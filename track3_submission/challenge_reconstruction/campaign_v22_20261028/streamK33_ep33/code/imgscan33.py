"""K33-inhand (GT-free): per-frame image evidence for the spoon pose in a bundle, robust to hand occlusion. For roll offsets R (about the long
PCA axis through COM) render OUR mesh with the bundle camera_from_object and compare with OUR SAM2 spoon mask and OUR stereo depth:
  rec  = |mask & render| / |mask|            (visible spoon pixels must lie inside the render; occlusion-robust)
  dm   = mean px distance of mask pixels to the render silhouette (0 if inside)
  dz   = median signed (render - stereo) depth on mask&render, mm
usage: imgscan33.py BUNDLE_EP_DIR OUT.tsv [step]"""
import sys, json, numpy as np, cv2, trimesh
from multiprocessing import Pool
from scipy.spatial.transform import Rotation as Rot
B, OUT = sys.argv[1], sys.argv[2]; STEP = int(sys.argv[3]) if len(sys.argv) > 3 else 1
C = '/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/canonical/evaluation/episode_000033'
K = json.load(open(f'{C}/intrinsics.json')); fx, fy, cx, cy, W, H = K['fx'], K['fy'], K['cx'], K['cy'], K['width'], K['height']
z = np.load(f'{B}/object_poses.npz', allow_pickle=False); Tc = z['camera_from_object'][:, list(z['names']).index('wooden_spoon')]
m = trimesh.load(f'{B}/meshes/wooden_spoon.obj', force='mesh'); v = np.asarray(m.vertices); f = np.asarray(m.faces); com = np.asarray(m.center_mass)
c0 = v - v.mean(0); w, U = np.linalg.eigh(c0.T @ c0 / len(c0)); long = U[:, 2]
if long[np.argmax(np.abs(long))] < 0: long = -long
ROLLS = [-60, -40, -30, -20, -10, 0, 10, 20, 30, 40, 60, 180]
def render(T):
    p = v @ T[:3, :3].T + T[:3, 3]; u = fx * p[:, 0] / p[:, 2] + cx; vv = fy * p[:, 1] / p[:, 2] + cy
    zb = np.full((H, W), np.inf, np.float32); fz = p[f, 2].mean(1)
    for t in np.argsort(-fz):
        pts = np.stack([u[f[t]], vv[f[t]]], 1).round().astype(np.int32)
        cv2.fillConvexPoly(zb, pts, float(fz[t]))
    return zb
def one(t):
    mask = cv2.imread(f'{C}/objects/wooden_spoon/masks/1/{t:06d}.png', cv2.IMREAD_UNCHANGED) > 0
    dv = cv2.imread(f'{C}/depth/{t:06d}.png', cv2.IMREAD_UNCHANGED).astype(np.float64); dz = np.where(dv < 65535, 65535 / np.maximum(dv, 1) - 1, np.nan)
    row = [t, int(mask.sum())]
    for r in ROLLS:
        E = Rot.from_rotvec(long * np.radians(r)).as_matrix(); Bm = np.eye(4); Bm[:3, :3] = E; Bm[:3, 3] = com - E @ com
        zb = render(Tc[t] @ Bm); rm = np.isfinite(zb)
        if mask.sum() < 30: row += [np.nan] * 3; continue
        rec = (rm & mask).sum() / mask.sum()
        dt = cv2.distanceTransform((~rm).astype(np.uint8), cv2.DIST_L2, 3); dm = dt[mask].mean()
        both = rm & mask & np.isfinite(dz); sdz = np.nanmedian(zb[both] - dz[both]) * 1000 if both.sum() > 30 else np.nan
        row += [rec, dm, sdz]
    return row
if __name__ == '__main__':
    with Pool(int(__import__('os').environ.get('SLURM_CPUS_PER_TASK', 8))) as P: rows = P.map(one, range(0, len(Tc), STEP))
    hdr = 'frame\tmaskpx\t' + '\t'.join(f'rec{r}\tdm{r}\tdz{r}' for r in ROLLS)
    open(OUT, 'w').write(hdr + '\n' + '\n'.join('\t'.join(f'{x:.3f}' if isinstance(x, float) else str(x) for x in r) for r in rows) + '\n')
    print('done', len(rows))
