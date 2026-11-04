"""GT-free image evidence for the ep35 static-window spoon roll: for frames F and roll offsets r (body roll about the long PCA axis through
the COM, same convention as build_h.py; H1 = -14), render our mesh with the bundle camera_from_object and compare with OUR SAM2 mask (IoU),
OUR stereo depth (median |dz| on mask&render) and OUR 2D gray-ball detections (pooled GT-free ball centres, median nearest px).
usage: imgev.py BUNDLE_EP_DIR f0,f1,..."""
import sys, json, numpy as np, cv2, trimesh
from scipy.spatial.transform import Rotation as Rot
B = sys.argv[1]; F = [int(x) for x in sys.argv[2].split(',')]
C = '/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/canonical/evaluation/episode_000035'
K = json.load(open(f'{C}/intrinsics.json')); fx, fy, cx, cy, W, H = K['fx'], K['fy'], K['cx'], K['cy'], K['width'], K['height']
z = np.load(f'{B}/object_poses.npz', allow_pickle=False); Tc = z['camera_from_object'][:, 0]
m = trimesh.load(f'{B}/meshes/wooden_spoon.obj', force='mesh'); parts = m.split(only_watertight=False); outer = max(parts, key=lambda p: p.area)
v = np.asarray(outer.vertices); f = np.asarray(outer.faces); com = np.asarray(m.center_mass)
c0 = np.asarray(m.vertices) - np.asarray(m.vertices).mean(0); w, U = np.linalg.eigh(c0.T @ c0 / len(c0)); long = U[:, 2]
if long[np.argmax(np.abs(long))] < 0: long = -long
bc = json.load(open('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/streamG18_spoonmesh/out/ballcenters.json'))
pe = bc['per_episode_centres']; balls = np.array([np.median([pe[e][k] for e in pe], 0) for k in ('near', 'far', 'tip')])
det = json.load(open(f'{C}/ball_detections_wooden_spoon.json'))['frames']
def render(T):
    p = v @ T[:3, :3].T + T[:3, 3]; u = fx * p[:, 0] / p[:, 2] + cx; vv = fy * p[:, 1] / p[:, 2] + cy
    zb = np.full((H, W), np.inf, np.float32); order = np.argsort(-p[f].mean(1)[:, 2])
    for t in order:
        pts = np.stack([u[f[t]], vv[f[t]]], 1).round().astype(np.int32); tmp = np.zeros((H, W), np.uint8)
        x0, y0 = pts.min(0); x1, y1 = pts.max(0)
        if x1 < 0 or y1 < 0 or x0 >= W or y0 >= H: continue
        cv2.fillConvexPoly(zb, pts, float(p[f[t], 2].mean()))
    return zb
rolls = np.arange(-24, 13, 4.0); res = {r: [] for r in rolls}
for t in F:
    mask = cv2.imread(f'{C}/objects/wooden_spoon/masks/1/{t:06d}.png', cv2.IMREAD_UNCHANGED) > 0
    dv = cv2.imread(f'{C}/depth/{t:06d}.png', cv2.IMREAD_UNCHANGED).astype(np.float64); dz = np.where(dv < 65535, 65535 / np.maximum(dv, 1) - 1, np.nan)
    dets = np.array([d['uv'] for d in det[str(t)]]) if isinstance(det, dict) and str(t) in det else (np.array([d['uv'] for d in det[t]]) if isinstance(det, list) and len(det) > t and det[t] else np.zeros((0, 2)))
    for r in rolls:
        E = Rot.from_rotvec(long * np.radians(r)).as_matrix(); Bm = np.eye(4); Bm[:3, :3] = E; Bm[:3, 3] = com - E @ com; T = Tc[t] @ Bm
        zb = render(T); rm = np.isfinite(zb); iou = (rm & mask).sum() / max((rm | mask).sum(), 1)
        both = rm & mask & np.isfinite(dz); ddz = np.nanmedian(np.abs(zb[both] - dz[both])) * 1000 if both.sum() > 50 else np.nan
        pb = balls @ T[:3, :3].T + T[:3, 3]; ub = np.stack([fx * pb[:, 0] / pb[:, 2] + cx, fy * pb[:, 1] / pb[:, 2] + cy], 1)
        bpx = np.median(np.min(np.linalg.norm(ub[:, None] - dets[None], axis=2), 1)) if len(dets) >= 2 else np.nan
        res[r].append((iou, ddz, bpx))
print('frames', F)
for r in rolls:
    a = np.array(res[r], float); print(f'roll {r:+5.0f}: IoU {np.nanmean(a[:,0]):.4f}  |dz| {np.nanmean(a[:,1]):5.2f}mm  balls {np.nanmedian(a[:,2]):5.2f}px')
