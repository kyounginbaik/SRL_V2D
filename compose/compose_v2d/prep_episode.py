"""Per-episode data preparation for ComPose on the V2D ego rig.

Writes outputs/compose_ep{ep}_{obj}/
  rgb_undist/{f:05d}.jpg   undistorted pinhole RGB (ego_cam_a, native focal, centred pp) -> DINO crops + WiLoR
  mono/{f:05d}.jpg         rectified-left mono (reused from outputs/mono_frames_ep{ep} when present) -> stereo VO
  masks_rgb/{f:05d}.png    object mask on the undistorted RGB, transferred from the mono SAM3 mask via stereo depth
  hands_rgb/{f:05d}.png    same for the hand mask
  pc_cache.npz             pc_{f}: (M,6) float16 [xyz metres in rect-left(f) frame | mono intensity x3]
                           object mask minus 15px-dilated hand, close5, depth-connected core, erode5 (fuse_keyframes_icp recipe)
  meta.json                K_rect, K_a_undist, T_a_from_rectL, sizes, frame count
"""
import argparse, json, os, sys
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, backproject, cam_params, depth_connected_core, load_calib, load_depth, load_T_a_from_rectL, mono_masks_dir, split_of, video_path, work_dir
from rectify import EgoStereoRectifier


def fill_mask(m):
    """Close + fill enclosed holes of a splatted binary mask."""
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    h, w = m.shape
    ff = m.copy(); ff = np.pad(ff, 1); cv2.floodFill(ff, None, (0, 0), 2)
    holes = (ff[1:-1, 1:-1] == 0)
    m[holes] = 1
    return m > 0


def transfer_mask_to_rgb(mask_left, depth, K_rect, T_a, K_und, Wa, Ha):
    """Splat the mono mask's stereo points into the undistorted RGB at half resolution, fill, upsample."""
    P, _, _ = backproject(depth, mask_left, K_rect)
    out_h = np.zeros((Ha // 2, Wa // 2), np.uint8)
    if len(P) == 0:
        return out_h.repeat(2, 0).repeat(2, 1)[:Ha, :Wa] > 0
    Pa = P @ T_a[:3, :3].T + T_a[:3, 3]
    ok = Pa[:, 2] > 0.05
    x = K_und[0, 0] * Pa[ok, 0] / Pa[ok, 2] + K_und[0, 2]
    y = K_und[1, 1] * Pa[ok, 1] / Pa[ok, 2] + K_und[1, 2]
    xi = np.round(x / 2).astype(int); yi = np.round(y / 2).astype(int)
    ins = (xi >= 0) & (xi < Wa // 2) & (yi >= 0) & (yi < Ha // 2)
    out_h[yi[ins], xi[ins]] = 1
    out_h = fill_mask(out_h).astype(np.uint8)
    out = cv2.resize(out_h, (Wa, Ha), interpolation=cv2.INTER_NEAREST)
    return out > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ep', type=int, required=True)
    ap.add_argument('--obj', required=True)
    ap.add_argument('--max_points', type=int, default=5000)
    ap.add_argument('--jpg_q', type=int, default=95)
    ap.add_argument('--skip_rgb', action='store_true')
    ap.add_argument('--exclude_objs', nargs='*', default=[], help='other objects of the episode whose (dilated) masks are subtracted from this object mask (e.g. the lid on the pot)')
    a = ap.parse_args()
    ep, obj = a.ep, a.obj
    out = work_dir(ep, obj)
    for s in ('rgb_undist', 'mono', 'masks_rgb', 'hands_rgb'):
        (out / s).mkdir(exist_ok=True)

    calib = load_calib()
    rect = EgoStereoRectifier(calib, episode=ep)
    Ka, Da, (Wa, Ha) = cam_params(calib, 'ego_cam_a')
    K_und = np.array([[Ka[0, 0], 0, (Wa - 1) / 2], [0, Ka[1, 1], (Ha - 1) / 2], [0, 0, 1]])
    depth_all, dframes, dmeta = load_depth(ep)
    K_rect = np.array(dmeta['K_rect'])
    T_a, T_a_src = load_T_a_from_rectL(ep, dmeta)
    n = len(dframes)
    print(f'[ep{ep} {obj}] {n} depth frames; T_a source = {T_a_src}; RGB {Wa}x{Ha}')

    # ── RGB: decode sequentially, undistort, save ──
    if not a.skip_rgb:
        mu1, mu2 = cv2.initUndistortRectifyMap(Ka, Da, None, K_und, (Wa, Ha), cv2.CV_32FC1)
        cap = cv2.VideoCapture(str(video_path(ep, 'a'))); i = 0
        while True:
            ok, fr = cap.read()
            if not ok: break
            cv2.imwrite(str(out / 'rgb_undist' / f'{i:05d}.jpg'), cv2.remap(fr, mu1, mu2, cv2.INTER_LINEAR),
                        [cv2.IMWRITE_JPEG_QUALITY, a.jpg_q])
            i += 1
        cap.release(); print(f'  RGB frames written: {i}')

    # ── mono rect-left ──
    src_mono = ROOT / 'outputs' / f'mono_frames_ep{ep}'
    if src_mono.exists() and len(list(src_mono.glob('*.jpg'))) >= n:
        for f in range(n):
            dst = out / 'mono' / f'{f:05d}.jpg'
            if not dst.exists():
                os.symlink(src_mono / f'{f:05d}.jpg', dst)
        print(f'  mono: linked {n} frames from {src_mono}')
    else:
        capb = cv2.VideoCapture(str(video_path(ep, 'b'))); capc = cv2.VideoCapture(str(video_path(ep, 'c'))); i = 0
        while True:
            okb, fb = capb.read(); okc, fc = capc.read()
            if not (okb and okc): break
            L, _ = rect.rectify(cv2.cvtColor(fc, cv2.COLOR_BGR2GRAY), cv2.cvtColor(fb, cv2.COLOR_BGR2GRAY))
            cv2.imwrite(str(out / 'mono' / f'{i:05d}.jpg'), L, [cv2.IMWRITE_JPEG_QUALITY, 95]); i += 1
        print(f'  mono: rectified {i} frames')

    # ── masks + point clouds ──
    md = mono_masks_dir(ep, obj)
    rng = np.random.default_rng(0)
    pcs = {}; stats = []
    for k, f in enumerate(dframes):
        depth = depth_all[k]
        mo = cv2.imread(str(md / f'obj_{f:05d}.png'), 0); mh = cv2.imread(str(md / f'hand_{f:05d}.png'), 0)
        if mo is None:
            stats.append(dict(frame=f, obj_px=0, n_pts=0)); continue
        mo = mo > 0; mh = (mh > 0) if mh is not None else np.zeros_like(mo)
        for other in a.exclude_objs:
            mx = cv2.imread(str(mono_masks_dir(ep, other) / f'obj_{f:05d}.png'), 0)
            if mx is not None and mx.any():
                keep = mo & ~(cv2.dilate((mx > 0).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0)
                if keep.sum() >= max(1500, 0.15 * mo.sum()): mo = keep      # unless the other mask swallows this object (pot mask on the lid): then don't subtract
        # RGB-side masks (for DINO crop / IoU gating)
        m_rgb = transfer_mask_to_rgb(mo, depth, K_rect, T_a, K_und, Wa, Ha)
        h_rgb = transfer_mask_to_rgb(mh, depth, K_rect, T_a, K_und, Wa, Ha) if mh.any() else np.zeros((Ha, Wa), bool)
        cv2.imwrite(str(out / 'masks_rgb' / f'{f:05d}.png'), m_rgb.astype(np.uint8) * 255)
        cv2.imwrite(str(out / 'hands_rgb' / f'{f:05d}.png'), h_rgb.astype(np.uint8) * 255)
        # hand-free object core -> stereo points
        hd = cv2.dilate(mh.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
        m = mo & ~hd
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool)
        if m.sum() >= 50:
            m = depth_connected_core(m, depth)
        core = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & (depth > 0)
        P, v, u = backproject(depth, core, K_rect)
        mono = cv2.imread(str(out / 'mono' / f'{f:05d}.jpg'), 0)
        inten = mono[v, u].astype(np.float32) / 255.0 if mono is not None else np.zeros(len(v), np.float32)
        if len(P) > a.max_points:
            idx = rng.choice(len(P), a.max_points, replace=False); P = P[idx]; inten = inten[idx]
        if len(P) >= 10:
            pcs[f'pc_{f}'] = np.concatenate([P, np.repeat(inten[:, None], 3, 1)], 1).astype(np.float16)
        stats.append(dict(frame=f, obj_px=int(mo.sum()), hand_px=int(mh.sum()), core_px=int(core.sum()), n_pts=int(len(P)),
                          rgb_px=int(m_rgb.sum()), z_med=float(np.median(P[:, 2])) if len(P) else None))
        if k % 50 == 0:
            print(f'  frame {f}: obj {int(mo.sum())} px, core {int(core.sum())} px -> {len(P)} pts, rgb mask {int(m_rgb.sum())} px')
    np.savez_compressed(out / 'pc_cache.npz', **pcs)
    meta = dict(ep=ep, obj=obj, split=split_of(ep), n_frames=n, frames=dframes, K_rect=K_rect.tolist(), K_a_undist=K_und.tolist(),
                K_a=Ka.tolist(), D_a=Da.tolist(), rgb_size_wh=[Wa, Ha], mono_size_wh=[1280, 800], T_a_from_rectL=T_a.tolist(),
                T_a_source=T_a_src, R1_rect_from_camc=rect.R1.tolist(), baseline_m=rect.baseline, mono_masks_dir=str(md), stats=stats)
    json.dump(meta, open(out / 'meta.json', 'w'), indent=1)
    npts = [s['n_pts'] for s in stats]
    print(f'  pc_cache: {len(pcs)}/{n} frames with points, median {int(np.median(npts))} pts; saved {out}')


if __name__ == '__main__':
    main()
