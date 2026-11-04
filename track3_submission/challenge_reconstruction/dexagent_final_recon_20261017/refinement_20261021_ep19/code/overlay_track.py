"""Overlay the reconstructed track pose of one object (filled mesh, green) on the real RGB at given times.
usage: overlay_track.py EPISODE OBJECT OUT.jpg t1 t2 ..."""
import sys, numpy as np, cv2, trimesh
ep, obj, out = int(sys.argv[1]), sys.argv[2], sys.argv[3]; ts = [float(x) for x in sys.argv[4:]]
D = f'/data1/SRL_V2D/challenge_reconstruction/final_recon/episode_{ep:06d}'
z = np.load(D + '/object_poses.npz'); names = [str(x) for x in z['names']]; io = names.index(obj)
cam = np.load(D + '/camera_motion.npz'); K = cam['K'].copy(); K[:2] *= 2.0
m = trimesh.load(f'{D}/assets/{obj}_visual.obj', force='mesh'); V = np.asarray(m.vertices)
F = f'/data1/SRL_V2D/challenge_submissions/track3/eval_hawor_latest_20261008/outputs/episode_{ep:06d}/{obj}/frames/%06d.png'
tiles = []
for t in ts:
    k = int(round(t * 20)); im = cv2.imread(F % k)
    Tc = np.linalg.inv(z['world_from_rgb'][k]) @ z['world_from_object'][k, io]; P = V @ Tc[:3, :3].T + Tc[:3, 3]; uv = P @ K.T; uv = uv[:, :2] / uv[:, 2:]
    ov = im.copy()
    for f in m.faces[::2]: cv2.polylines(ov, [np.round(uv[f]).astype(np.int32)], True, (0, 255, 0), 1)
    cu, cv_ = np.median(uv, 0).astype(int); x0, y0 = max(cu - 320, 0), max(cv_ - 280, 0)
    def crop(a):
        c = a[y0:y0 + 560, x0:x0 + 640].copy(); return cv2.copyMakeBorder(c, 0, 560 - c.shape[0], 0, 640 - c.shape[1], cv2.BORDER_CONSTANT)
    a, b = crop(im), crop(cv2.addWeighted(ov, 0.6, im, 0.4, 0)); cv2.putText(a, f't={t:.2f}s', (10, 40), 0, 1.2, (0, 255, 255), 3)
    tiles.append(np.hstack([a, b]))
cv2.imwrite(out, cv2.resize(np.vstack(tiles), None, fx=0.5, fy=0.5)); print('wrote', out)
