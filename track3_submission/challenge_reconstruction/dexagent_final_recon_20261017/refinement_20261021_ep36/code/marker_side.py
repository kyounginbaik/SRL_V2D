"""Draw where marker-like points on the local -Z face (handle centreline, 1.5 cm off the face) land for the track pose (green)
and the 180-deg roll twin (red) on the real RGB, to compare with the grey mocap balls. usage: marker_side.py OUT t1 t2 ..."""
import sys, numpy as np, cv2
from scipy.spatial.transform import Rotation as R
out, ts = sys.argv[1], [float(x) for x in sys.argv[2:]]
D = '/data1/SRL_V2D/challenge_reconstruction/final_recon/episode_000036'; z = np.load(D + '/object_poses.npz'); K = np.load(D + '/camera_motion.npz')['K'].copy(); K[:2] *= 2
F = '/data1/SRL_V2D/challenge_submissions/track3/eval_hawor_latest_20261008/outputs/episode_000036/wooden_spoon/frames/%06d.png'
pts = np.array([[x, 0, -0.023] for x in np.arange(-0.10, 0.04, 0.01)])   # -Z face + stalk
axis = np.array([[x, 0, 0] for x in np.arange(-0.16, 0.145, 0.01)])
twin = R.from_euler('x', 180, degrees=True).as_matrix(); tiles = []
for t in ts:
    k = int(round(t * 20)); im = cv2.imread(F % k); Tc = np.linalg.inv(z['world_from_rgb'][k]) @ z['world_from_object'][k, 0]
    def proj(Q, Rm): X = Q @ Rm.T + Tc[:3, 3]; uv = X @ K.T; return (uv[:, :2] / uv[:, 2:]).astype(int)
    for q in proj(axis, Tc[:3, :3]): cv2.circle(im, tuple(q), 3, (255, 255, 0), -1)
    for q in proj(pts, Tc[:3, :3]): cv2.circle(im, tuple(q), 6, (0, 255, 0), 2)
    for q in proj(pts, Tc[:3, :3] @ twin): cv2.circle(im, tuple(q), 6, (0, 0, 255), 2)
    c = proj(axis, Tc[:3, :3]).mean(0).astype(int); x0, y0 = max(c[0] - 330, 0), max(c[1] - 330, 0)
    cr = im[y0:y0 + 660, x0:x0 + 660].copy(); cr = cv2.copyMakeBorder(cr, 0, 660 - cr.shape[0], 0, 660 - cr.shape[1], cv2.BORDER_CONSTANT)
    cv2.putText(cr, f't={t:.1f}s', (10, 40), 0, 1.3, (255, 255, 255), 3); tiles.append(cr)
rows = [np.hstack(tiles[i:i + 4] + [np.zeros_like(tiles[0])] * (4 - len(tiles[i:i + 4]))) for i in range(0, len(tiles), 4)]
cv2.imwrite(out, cv2.resize(np.vstack(rows), None, fx=0.5, fy=0.5)); print('wrote', out)
