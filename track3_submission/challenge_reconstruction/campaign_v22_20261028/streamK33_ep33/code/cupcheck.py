"""K33 GT-free cup frame-0 check (the kit's scene translation offset in ep33 comes from slot 0 = blue_cup's frame-0 position):
render our cup mesh at the v14 camera_from_object pose on the static start frames with camera-frame shifts, compare with OUR SAM2 cup mask
(IoU) and OUR stereo depth (median |dz|). usage: cupcheck.py BUNDLE_EP_DIR f0,f1,..."""
import sys, json, numpy as np, cv2, trimesh
B = sys.argv[1]; F = [int(x) for x in sys.argv[2].split(',')]
C = '/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/canonical/evaluation/episode_000033'
K = json.load(open(f'{C}/intrinsics.json')); fx, fy, cx, cy, W, H = K['fx'], K['fy'], K['cx'], K['cy'], K['width'], K['height']
z = np.load(f'{B}/object_poses.npz', allow_pickle=False); Tc = z['camera_from_object'][:, list(z['names']).index('blue_cup')]
m = trimesh.load(f'{B}/meshes/blue_cup.obj', force='mesh'); v = np.asarray(m.vertices); f = np.asarray(m.faces)
def render(T):
    p = v @ T[:3, :3].T + T[:3, 3]; u = fx * p[:, 0] / p[:, 2] + cx; vv = fy * p[:, 1] / p[:, 2] + cy
    zb = np.full((H, W), np.inf, np.float32)
    for t in np.argsort(-p[f].mean(1)[:, 2]):
        pts = np.stack([u[f[t]], vv[f[t]]], 1).round().astype(np.int32); cv2.fillConvexPoly(zb, pts, float(p[f[t], 2].mean()))
    return zb
shifts = [(0, 0, 0)] + [(d, 0, 0) for d in (-0.02, 0.02)] + [(0, d, 0) for d in (-0.02, 0.02)] + [(0, 0, d) for d in (-0.03, -0.015, 0.015, 0.03)]
for s in shifts:
    out = []
    for t in F:
        mask = cv2.imread(f'{C}/objects/blue_cup/masks/1/{t:06d}.png', cv2.IMREAD_UNCHANGED) > 0
        dv = cv2.imread(f'{C}/depth/{t:06d}.png', cv2.IMREAD_UNCHANGED).astype(np.float64); dz = np.where(dv < 65535, 65535 / np.maximum(dv, 1) - 1, np.nan)
        T = Tc[t].copy(); T[:3, 3] += np.array(s); zb = render(T); rm = np.isfinite(zb)
        iou = (rm & mask).sum() / max((rm | mask).sum(), 1); both = rm & mask & np.isfinite(dz)
        out.append((iou, np.nanmedian(zb[both] - dz[both]) * 1000 if both.sum() > 50 else np.nan, np.nanmedian(np.abs(zb[both] - dz[both])) * 1000 if both.sum() > 50 else np.nan))
    a = np.array(out); print(f'shift cam {s}: IoU {a[:,0].mean():.3f}  signed dz {np.nanmean(a[:,1]):+6.1f}mm  |dz| {np.nanmean(a[:,2]):5.1f}mm', flush=True)
