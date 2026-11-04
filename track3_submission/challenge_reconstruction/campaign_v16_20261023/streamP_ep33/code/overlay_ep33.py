"""Stream P ep33: overlay the reference cup (blue wireframe points) and spoon (yellow) projected into the source RGB at given
times, plus the cup axis (red line from floor to opening). usage: overlay_ep33.py OUT.jpg [--poses NPZ] t1 t2 ..."""
import sys, numpy as np, cv2, trimesh
FR = '/data1/SRL_V2D/challenge_reconstruction/final_recon/episode_000033'
F = '/data1/SRL_V2D/challenge_submissions/track3/eval_hawor_latest_20261008/outputs/episode_000033/blue_cup/frames/%06d.png'
args = sys.argv[2:]; poses = FR + '/object_poses.npz'
if '--poses' in args: i = args.index('--poses'); poses = args[i + 1]; del args[i:i + 2]
ts = [float(x) for x in args]; z = np.load(poses); names = [str(x) for x in z['names']]
K = np.load(FR + '/camera_motion.npz')['K'].copy(); K[:2] *= 2
cup = trimesh.load(FR + '/assets/blue_cup_visual.obj', force='mesh'); sp = trimesh.load(FR + '/assets/wooden_spoon_visual.obj', force='mesh')
cv_ = cup.vertices[::2]; sv = sp.vertices[::3]; axis = np.array([[0, 0, z_] for z_ in np.linspace(0.0875, -0.083, 30)])
tiles = []
for t in ts:
    k = int(round(t * 20)); im = cv2.imread(F % k); C = np.linalg.inv(z['world_from_rgb'][k])
    def proj(Q, T):
        X = Q @ (C @ T)[:3, :3].T + (C @ T)[:3, 3]; q = X @ K.T; return (q[:, :2] / q[:, 2:]).astype(int)
    for q in proj(cv_, z['world_from_object'][k, names.index('blue_cup')]): cv2.circle(im, tuple(q), 2, (255, 120, 0), -1)
    for q in proj(sv, z['world_from_object'][k, names.index('wooden_spoon')]): cv2.circle(im, tuple(q), 2, (0, 230, 255), -1)
    a = proj(axis, z['world_from_object'][k, names.index('blue_cup')]); cv2.line(im, tuple(a[0]), tuple(a[-1]), (0, 0, 255), 4)
    c = a.mean(0).astype(int); x0, y0 = int(np.clip(c[0] - 400, 0, im.shape[1] - 800)), int(np.clip(c[1] - 400, 0, im.shape[0] - 800))
    cr = im[y0:y0 + 800, x0:x0 + 800].copy(); cv2.putText(cr, f't={t:.2f}s', (10, 50), 0, 1.6, (255, 255, 255), 4); tiles.append(cr)
while len(tiles) % 4: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(sys.argv[1], cv2.resize(np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]), None, fx=0.45, fy=0.45)); print('wrote', sys.argv[1])
