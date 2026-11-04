"""Project the spoon (track pose and its 180-deg roll twin about local long X) into the real RGB; draw the bowl-opening axis
(local -Z) so the bowl-up/down branch can be judged. usage: overlay_spoon.py OUT.jpg t1 t2 ... [--poses object_poses.npz]"""
import sys, numpy as np, cv2, trimesh
from scipy.spatial.transform import Rotation as R
args = sys.argv[1:]; poses = '/data1/SRL_V2D/challenge_reconstruction/final_recon/episode_000036/object_poses.npz'
if '--poses' in args: i = args.index('--poses'); poses = args[i + 1]; args = args[:i] + args[i + 2:]
out, ts = args[0], [float(x) for x in args[1:]]
D = '/data1/SRL_V2D/challenge_reconstruction/final_recon/episode_000036'
z = np.load(poses); cam = np.load(D + '/camera_motion.npz'); K = cam['K'].copy(); K[:2] *= 2.0
m = trimesh.load(D + '/assets/wooden_spoon_visual.obj', force='mesh'); V = np.asarray(m.vertices)
ext = V.max(0) - V.min(0); print('local extents cm', np.round(ext * 100, 1))
bowl_end = V[:, 0] > np.median(V[:, 0]); c_b = V[np.argsort(V[:, 0])[-max(50, len(V)//6):]].mean(0)  # assume +X end; checked visually
F = '/data1/SRL_V2D/challenge_submissions/track3/eval_hawor_latest_20261008/outputs/episode_000036/wooden_spoon/frames/%06d.png'
twin = np.eye(4); twin[:3, :3] = R.from_euler('x', 180, degrees=True).as_matrix()
tiles = []
for t in ts:
    k = int(round(t * 20)); im = cv2.imread(F % k); Two = z['world_from_object'][k, 0]; Wrc = z['world_from_rgb'][k]
    crops = []
    for tag, T, col in (('track', Two, (0, 255, 0)), ('twin', Two @ twin, (0, 0, 255))):
        Tc = np.linalg.inv(Wrc) @ T; P = V @ Tc[:3, :3].T + Tc[:3, 3]; uv = (P @ K.T); uv = uv[:, :2] / uv[:, 2:]
        img = im.copy(); ov = img.copy()
        for f in m.faces[::2]:
            cv2.fillConvexPoly(ov, np.round(uv[f]).astype(np.int32), col)
        img = cv2.addWeighted(ov, 0.35, img, 0.65, 0)
        # bowl opening axis: local -Z at the +X half centroid and at the -X half
        for end, cc in ((V[V[:, 0] > np.percentile(V[:, 0], 80)].mean(0), (255, 255, 0)), (V[V[:, 0] < np.percentile(V[:, 0], 20)].mean(0), (255, 0, 255))):
            a = Tc[:3, :3] @ end + Tc[:3, 3]; b = Tc[:3, :3] @ (end + np.array([0, 0, -0.04])) + Tc[:3, 3]
            pa = (K @ a)[:2] / a[2]; pb = (K @ b)[:2] / b[2]; cv2.arrowedLine(img, tuple(map(int, pa)), tuple(map(int, pb)), cc, 4, tipLength=0.3)
        cu, cv_ = np.median(uv, 0).astype(int); x0, y0 = max(cu - 300, 0), max(cv_ - 260, 0)
        cr = img[y0:y0 + 520, x0:x0 + 600].copy(); cr = cv2.copyMakeBorder(cr, 0, 520 - cr.shape[0], 0, 600 - cr.shape[1], cv2.BORDER_CONSTANT)
        cv2.putText(cr, f't={t:.1f}s {tag}', (10, 35), 0, 1.1, (255, 255, 255), 3); crops.append(cr)
    raw = im[y0:y0 + 520, x0:x0 + 600].copy(); raw = cv2.copyMakeBorder(raw, 0, 520 - raw.shape[0], 0, 600 - raw.shape[1], cv2.BORDER_CONSTANT)
    tiles.append(np.hstack([raw] + crops))
cv2.imwrite(out, cv2.resize(np.vstack(tiles), None, fx=0.5, fy=0.5)); print('wrote', out)
