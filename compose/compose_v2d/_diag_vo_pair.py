"""Direct PnP between two arbitrary frames a -> b (static background), compared with the chained VO: cam_b_T_cam_a."""
import sys, os, cv2, numpy as np
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, load_depth, mono_masks_dir, work_dir
from stereo_vo import dyn_mask
ep, obj = int(sys.argv[1]), sys.argv[2]; pairs = [tuple(map(int, p.split(','))) for p in sys.argv[3:]]
depth_all, dframes, meta = load_depth(ep); K = np.array(meta['K_rect']); md = mono_masks_dir(ep, obj)
mf = ROOT / 'outputs' / f'mono_frames_ep{ep}'; E = np.load(work_dir(ep, obj) / 'ego_cam0_T_cam.npz')['cam0_T_cam']
sift = cv2.SIFT_create(4000); bf = cv2.BFMatcher()
def get(g):
    im = cv2.imread(str(mf / f'{g:05d}.jpg'), 0); static = (~dyn_mask(md, g, im.shape)).astype(np.uint8) * 255
    kp, des = sift.detectAndCompute(im, static); return kp, des, depth_all[dframes.index(g)].astype(np.float64) / 1000
for a, b in pairs:
    kp_a, des_a, d_a = get(a); kp_b, des_b, _ = get(b)
    good = [m for m, n in bf.knnMatch(des_a, des_b, k=2) if m.distance < 0.75 * n.distance]
    pa = np.array([kp_a[m.queryIdx].pt for m in good]); pb = np.array([kp_b[m.trainIdx].pt for m in good])
    z = d_a[np.round(pa[:, 1]).astype(int), np.round(pa[:, 0]).astype(int)]; ok = (z > 0.2) & (z < 6.0)
    P3 = np.stack([(pa[ok, 0] - K[0, 2]) * z[ok] / K[0, 0], (pa[ok, 1] - K[1, 2]) * z[ok] / K[1, 1], z[ok]], 1)
    succ, rv, tv, inl = cv2.solvePnPRansac(P3, pb[ok], K, None, reprojectionError=1.5, iterationsCount=5000, confidence=0.999, flags=cv2.SOLVEPNP_SQPNP)
    rv, tv = cv2.solvePnPRefineLM(P3[inl.ravel()], pb[ok][inl.ravel()], K, None, rv, tv)
    Rd = Rot.from_rotvec(rv.ravel()).as_matrix(); td = tv.ravel()                     # cam_b_T_cam_a direct
    Tc = np.linalg.inv(E[b]) @ E[a]                                                    # chained: cam_b_T_cam_a
    dR = np.degrees(np.linalg.norm(Rot.from_matrix(Rd @ Tc[:3, :3].T).as_rotvec())); dt = np.linalg.norm(td - Tc[:3, 3]) * 100
    print(f'ep{ep} {a}->{b}: matches {len(good)} inliers {len(inl)} | direct rot {np.degrees(np.linalg.norm(rv)):.2f} deg trans {np.linalg.norm(td)*100:.1f} cm | chained rot {np.degrees(np.linalg.norm(Rot.from_matrix(Tc[:3,:3]).as_rotvec())):.2f} deg trans {np.linalg.norm(Tc[:3,3])*100:.1f} cm | diff {dR:.2f} deg / {dt:.1f} cm')
