"""Estimate the 3 mocap-ball body coordinates from their pixels at a trusted frame (0.5 s, bowl up at rest), then predict
them for the track pose (green) and the roll twin (red) at later frames. usage: marker_predict.py OUT t1 t2 ..."""
import sys, numpy as np, cv2
from scipy.spatial.transform import Rotation as R
out, ts = sys.argv[1], [float(x) for x in sys.argv[2:] if not x.endswith('.npz')]
D = '/data1/SRL_V2D/challenge_reconstruction/final_recon/episode_000036'; z = np.load(sys.argv[-1] if sys.argv[-1].endswith('.npz') else D + '/object_poses.npz'); K = np.load(D + '/camera_motion.npz')['K'].copy(); K[:2] *= 2
F = '/data1/SRL_V2D/challenge_submissions/track3/eval_hawor_latest_20261008/outputs/episode_000036/wooden_spoon/frames/%06d.png'
def Tc(k): return np.linalg.inv(z['world_from_rgb'][k]) @ z['world_from_object'][k, 0]
axis = np.array([[x, 0, 0] for x in np.arange(-0.16, 0.145, 0.01)])
k0 = 10; T0 = Tc(k0); X = axis @ T0[:3, :3].T + T0[:3, 3]; uv = X @ K.T; c = (uv[:, :2] / uv[:, 2:]).mean(0).astype(int); x0, y0 = max(c[0] - 330, 0), max(c[1] - 330, 0)
tile = [(130, 158), (178, 170), (295, 121)]; px = [(x0 + 2 * a, y0 + 2 * b) for a, b in tile]
Kinv = np.linalg.inv(K); Rb = T0[:3, :3]; tb = T0[:3, 3]; body = []
for u, v in px:
    d = Rb.T @ (Kinv @ np.array([u, v, 1.0])); o = Rb.T @ (-tb)   # ray in body frame
    # closest approach of the ray to the body x-axis line
    e = np.array([1.0, 0, 0]); w = o; a_, b_, c_ = d @ d, d @ e, e @ e; dd, ee = d @ w, e @ w; den = a_ * c_ - b_ * b_
    s = (b_ * ee - c_ * dd) / den; tt = (a_ * ee - b_ * dd) / den; p = o + s * d; body.append(p)
body = np.array(body); print('marker body coords (cm):', np.round(body * 100, 1).tolist())
twin = R.from_euler('x', 180, degrees=True).as_matrix(); tiles = []
for t in ts:
    k = int(round(t * 20)); im = cv2.imread(F % k); T = Tc(k)
    def proj(Q, Rm): Xc = Q @ Rm.T + T[:3, 3]; q = Xc @ K.T; return (q[:, :2] / q[:, 2:]).astype(int)
    for q in proj(axis, T[:3, :3]): cv2.circle(im, tuple(q), 3, (255, 255, 0), -1)
    for q in proj(body, T[:3, :3]): cv2.circle(im, tuple(q), 16, (0, 255, 0), 4)
    for q in proj(body, T[:3, :3] @ twin): cv2.circle(im, tuple(q), 16, (0, 0, 255), 4)
    cc = proj(axis, T[:3, :3]).mean(0).astype(int); xa, ya = max(cc[0] - 330, 0), max(cc[1] - 330, 0)
    cr = im[ya:ya + 660, xa:xa + 660].copy(); cr = cv2.copyMakeBorder(cr, 0, 660 - cr.shape[0], 0, 660 - cr.shape[1], cv2.BORDER_CONSTANT)
    cv2.putText(cr, f't={t:.1f}s', (10, 40), 0, 1.3, (255, 255, 255), 3); tiles.append(cr)
rows = [np.hstack(tiles[i:i + 4] + [np.zeros_like(tiles[0])] * (4 - len(tiles[i:i + 4]))) for i in range(0, len(tiles), 4)]
cv2.imwrite(out, cv2.resize(np.vstack(rows), None, fx=0.5, fy=0.5)); np.save(out.replace('.jpg', '_body.npy'), body); print('wrote', out)
