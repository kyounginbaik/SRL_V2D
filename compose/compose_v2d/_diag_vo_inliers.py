"""Draw the PnP inliers of one VO step (f-1 -> f) on the rect-left frame to see what the ego-motion is locking onto."""
import sys, os, cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, load_depth, mono_masks_dir, work_dir
from stereo_vo import dyn_mask
ep, obj, f = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
depth_all, dframes, meta = load_depth(ep); K = np.array(meta['K_rect']); md = mono_masks_dir(ep, obj)
mf = ROOT / 'outputs' / f'mono_frames_ep{ep}'
sift = cv2.SIFT_create(4000); bf = cv2.BFMatcher()
def get(g):
    im = cv2.imread(str(mf / f'{g:05d}.jpg'), 0); static = (~dyn_mask(md, g, im.shape)).astype(np.uint8) * 255
    kp, des = sift.detectAndCompute(im, static); return im, kp, des, depth_all[dframes.index(g)].astype(np.float64) / 1000
im_a, kp_a, des_a, d_a = get(f - 1); im_b, kp_b, des_b, _ = get(f)
good = [m for m, n in bf.knnMatch(des_a, des_b, k=2) if m.distance < 0.75 * n.distance]
pa = np.array([kp_a[m.queryIdx].pt for m in good]); pb = np.array([kp_b[m.trainIdx].pt for m in good])
z = d_a[np.round(pa[:, 1]).astype(int), np.round(pa[:, 0]).astype(int)]; ok = (z > 0.2) & (z < 6.0)
P3 = np.stack([(pa[ok, 0] - K[0, 2]) * z[ok] / K[0, 0], (pa[ok, 1] - K[1, 2]) * z[ok] / K[1, 1], z[ok]], 1)
succ, rv, tv, inl = cv2.solvePnPRansac(P3, pb[ok], K, None, reprojectionError=1.5, iterationsCount=2000, confidence=0.999, flags=cv2.SOLVEPNP_SQPNP)
inl = inl.ravel(); pbo = pb[ok]; zo = z[ok]
print(f'ep{ep} {obj} step {f-1}->{f}: matches {len(good)}, with depth {ok.sum()}, inliers {len(inl)}; inlier depth percentiles 10/50/90: {np.percentile(zo[inl],[10,50,90]).round(2)}; rot {np.degrees(np.linalg.norm(rv)):.2f} deg, trans {np.linalg.norm(tv)*100:.2f} cm')
vis = cv2.cvtColor(im_b, cv2.COLOR_GRAY2BGR); dm = dyn_mask(md, f, im_b.shape)
vis[dm] = (vis[dm] * 0.5 + np.array([0, 0, 120])).astype(np.uint8)
for i in range(len(pbo)):
    c = (0, 255, 0) if i in set(inl.tolist()) else (0, 0, 255); cv2.circle(vis, tuple(np.round(pbo[i]).astype(int)), 4, c, -1)
out = ROOT / f'outputs/_diag_vo_ep{ep}_{obj}_f{f}.jpg'; cv2.imwrite(str(out), vis); print('wrote', out)
