"""Camera ego-motion on the rectified-left mono stream: chained frame-to-frame PnP on the STATIC background.

For each f: SIFT matches (f-1 -> f) outside the dilated object+hand masks, 3D from the stereo depth of f-1,
PnP-RANSAC + LM  ->  cam_f_T_cam_{f-1};  chained to cam0_T_cam(f).  Optionally a second, longer edge (f-k -> f)
is estimated and the two are fused (rotation: geodesic mean, translation: mean) to damp per-step drift.
Writes outputs/compose_ep{ep}_{obj}/ego_cam0_T_cam.npz  (cam0_T_cam (N,4,4), ok (N,), n_inl (N,))
and compares with outputs/egomotion_ep{ep}/ref_cam0_T_cam_kf.json when present.
"""
import argparse, json, os, sys
import cv2, numpy as np
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, load_depth, load_meta, mono_masks_dir, work_dir


def dyn_mask(md, f, shape, dil=31):
    mo = cv2.imread(str(md / f'obj_{f:05d}.png'), 0); mh = cv2.imread(str(md / f'hand_{f:05d}.png'), 0)
    m = np.zeros(shape, bool)
    if mo is not None: m |= mo > 0
    if mh is not None: m |= mh > 0
    return cv2.dilate(m.astype(np.uint8), np.ones((dil, dil), np.uint8)) > 0


def pnp_edge(kp_a, des_a, depth_a, kp_b, des_b, K, bf, reproj=1.5):
    """cam_b_T_cam_a from 3D (frame a, stereo depth) -> 2D (frame b)."""
    if des_a is None or des_b is None or len(kp_a) < 8 or len(kp_b) < 8:
        return None, 0
    good = [m for m, n in bf.knnMatch(des_a, des_b, k=2) if m.distance < 0.75 * n.distance]
    if len(good) < 12:
        return None, len(good)
    pa = np.array([kp_a[m.queryIdx].pt for m in good]); pb = np.array([kp_b[m.trainIdx].pt for m in good])
    z = depth_a[np.round(pa[:, 1]).astype(int), np.round(pa[:, 0]).astype(int)]
    ok = (z > 0.2) & (z < 6.0)
    if ok.sum() < 12:
        return None, int(ok.sum())
    P3 = np.stack([(pa[ok, 0] - K[0, 2]) * z[ok] / K[0, 0], (pa[ok, 1] - K[1, 2]) * z[ok] / K[1, 1], z[ok]], 1)
    succ, rv, tv, inl = cv2.solvePnPRansac(P3, pb[ok], K, None, reprojectionError=reproj, iterationsCount=2000,
                                           confidence=0.999, flags=cv2.SOLVEPNP_SQPNP)
    if not succ or inl is None or len(inl) < 12:
        return None, 0 if inl is None else len(inl)
    rv, tv = cv2.solvePnPRefineLM(P3[inl[:, 0]], pb[ok][inl[:, 0]], K, None, rv, tv)
    T = np.eye(4); T[:3, :3] = cv2.Rodrigues(rv)[0]; T[:3, 3] = tv.ravel()
    return T, len(inl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ep', type=int, required=True); ap.add_argument('--obj', required=True)
    ap.add_argument('--long_edge', type=int, default=5, help='also estimate (f-k -> f) and fuse; 0 = chain only')
    ap.add_argument('--nfeatures', type=int, default=8000)
    a = ap.parse_args()
    out = work_dir(a.ep, a.obj); meta = load_meta(a.ep, a.obj)
    K = np.array(meta['K_rect']); md = mono_masks_dir(a.ep, a.obj)
    depth_all, dframes, _ = load_depth(a.ep); n = len(dframes)
    sift = cv2.SIFT_create(nfeatures=a.nfeatures); bf = cv2.BFMatcher(cv2.NORM_L2)

    feats = {}
    def get(f):
        if f not in feats:
            im = cv2.imread(str(out / 'mono' / f'{f:05d}.jpg'), 0)
            static = (~dyn_mask(md, f, im.shape)).astype(np.uint8) * 255
            kp, des = sift.detectAndCompute(im, static)
            feats[f] = (kp, des, depth_all[dframes.index(f)].astype(np.float32) / 1000.0)
        return feats[f]

    cam0_T = np.tile(np.eye(4), (n, 1, 1)); ok_flag = np.zeros(n, bool); ok_flag[0] = True; n_inl = np.zeros(n, int)
    for f in range(1, n):
        kpa, desa, da = get(f - 1); kpb, desb, _ = get(f)
        T_f_fm1, ni = pnp_edge(kpa, desa, da, kpb, desb, K, bf)
        cands = []
        if T_f_fm1 is not None:
            cands.append(cam0_T[f - 1] @ np.linalg.inv(T_f_fm1)); n_inl[f] = ni
        k = a.long_edge
        if k > 0 and f - k >= 0 and ok_flag[f - k]:
            kpk, desk, dk = get(f - k)
            T_f_fmk, nk = pnp_edge(kpk, desk, dk, kpb, desb, K, bf)
            if T_f_fmk is not None and nk >= 30:
                cands.append(cam0_T[f - k] @ np.linalg.inv(T_f_fmk))
        if not cands:
            cam0_T[f] = cam0_T[f - 1]; ok_flag[f] = False
            print(f'  frame {f}: VO failed (inliers {ni}), holding previous pose'); continue
        if len(cands) == 1:
            cam0_T[f] = cands[0]
        else:
            # fuse: if the two disagree a lot, trust the short edge; else average
            dR = np.degrees(np.linalg.norm(Rot.from_matrix(cands[0][:3, :3].T @ cands[1][:3, :3]).as_rotvec()))
            dt = np.linalg.norm(cands[0][:3, 3] - cands[1][:3, 3])
            if dR > 1.0 or dt > 0.01:
                cam0_T[f] = cands[0]
            else:
                T = np.eye(4); T[:3, :3] = Rot.from_matrix(np.stack([c[:3, :3] for c in cands])).mean().as_matrix()
                T[:3, 3] = np.mean([c[:3, 3] for c in cands], 0); cam0_T[f] = T
        ok_flag[f] = True
        feats.pop(f - k - 1, None) if k > 0 else feats.pop(f - 2, None)
        if f % 50 == 0:
            mv = np.linalg.norm(cam0_T[f][:3, 3]) * 100; rv = np.degrees(np.linalg.norm(Rot.from_matrix(cam0_T[f][:3, :3]).as_rotvec()))
            print(f'  frame {f}: inliers {ni}, camera moved {mv:.1f} cm / {rv:.1f} deg since frame 0')
    np.savez(out / 'ego_cam0_T_cam.npz', cam0_T_cam=cam0_T, ok=ok_flag, n_inl=n_inl)
    print(f'saved {out / "ego_cam0_T_cam.npz"}: {ok_flag.sum()}/{n} frames ok')

    ref_p = ROOT / 'outputs' / f'egomotion_ep{a.ep}' / 'ref_cam0_T_cam_kf.json'
    if ref_p.exists():
        ref = {int(k): np.array(v) for k, v in json.load(open(ref_p)).items()}
        errs = []
        for f, Rr in sorted(ref.items()):
            if f == 0 or f >= n: continue
            P = cam0_T[f]
            dR = np.degrees(np.linalg.norm(Rot.from_matrix(P[:3, :3].T @ Rr[:3, :3]).as_rotvec())); dt = np.linalg.norm(P[:3, 3] - Rr[:3, 3]) * 100
            errs.append((dR, dt))
            print(f'  vs reference @ frame {f:3d}: rot err {dR:5.2f} deg, trans err {dt:5.2f} cm (camera moved {np.linalg.norm(Rr[:3,3])*100:.1f} cm / {np.degrees(np.linalg.norm(Rot.from_matrix(Rr[:3,:3]).as_rotvec())):.1f} deg)')
        e = np.array(errs); print(f'  median rot err {np.median(e[:,0]):.2f} deg, median trans err {np.median(e[:,1]):.2f} cm')


if __name__ == '__main__':
    main()
