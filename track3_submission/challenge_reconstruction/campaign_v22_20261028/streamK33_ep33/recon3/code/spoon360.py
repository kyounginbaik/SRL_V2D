"""K33-recon3 (GT-free): per-frame FULL-circle spoon roll scan about the long PCA axis through the COM (body frame), relative to a bundle's
camera_from_object, with three independent image cues (Jisu-style):
  sil  : two-sided occlusion-aware silhouette cost (FN + unoccluded FP)/|mask|, best over small image shifts (= camera x,y translation)
  dep  : stereo-depth residual of the visible spoon pixels at that best shift (mean |render z - stereo z|, mm, after removing the median
         offset; the median offset is reported separately) -> separates the concave from the convex side of the bowl
  ball : mean 3D distance (cm) of our stereo ball detections (inliers < 12 cm) to the nearest ball-model centre (G18 pooled centres)
Occluders: hand masks (2, 3), the other object's mask, and pixels whose stereo depth is > 1.5 cm in front of the render.
usage: spoon360.py BUNDLE_EP_DIR OUT.npz [STEP_DEG=10] [EPN=33]"""
import sys, os, json, numpy as np, cv2, trimesh
from multiprocessing import Pool
from scipy.spatial.transform import Rotation as Rot
B, OUT = sys.argv[1], sys.argv[2]; STEP = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0; EPN = int(sys.argv[4]) if len(sys.argv) > 4 else 33
C = f'/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/canonical/evaluation/episode_{EPN:06d}'
K = json.load(open(f'{C}/intrinsics.json')); fx, fy, cx, cy, W, H = K['fx'], K['fy'], K['cx'], K['cy'], K['width'], K['height']
OBJ = 'wooden_spoon'
z = np.load(f'{B}/object_poses.npz', allow_pickle=False); names = list(z['names']); Tc = z['camera_from_object'][:, names.index(OBJ)]
OTHER = ([n for n in names if n != OBJ] or [None])[0]
m = trimesh.load(f'{B}/meshes/{OBJ}.obj', force='mesh'); v = np.asarray(m.vertices); f = np.asarray(m.faces); com = np.asarray(m.center_mass)
c0 = v - v.mean(0); w, U = np.linalg.eigh(c0.T @ c0 / len(c0)); LONG = U[:, 2]
if LONG[np.argmax(np.abs(LONG))] < 0: LONG = -LONG
bc = json.load(open('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/streamG18_spoonmesh/out/ballcenters.json'))
pe = bc['per_episode_centres']; BM = np.array([np.median([pe[e][k] for e in pe], 0) for k in ('near', 'far', 'tip')])
BD = json.load(open(f'{C}/ball_detections_{OBJ}.json'))['frames']
ROLLS = np.arange(0, 360, STEP)
SH = np.arange(-15, 16, 3)

def rollmat(r):
    E = Rot.from_rotvec(LONG * np.radians(r)).as_matrix(); Bm = np.eye(4); Bm[:3, :3] = E; Bm[:3, 3] = com - E @ com; return Bm

def render(T):
    p = v @ T[:3, :3].T + T[:3, 3]; u = fx * p[:, 0] / p[:, 2] + cx; vv = fy * p[:, 1] / p[:, 2] + cy
    zb = np.full((H, W), np.inf, np.float32); fz = p[f, 2].mean(1)
    for t in np.argsort(-fz):
        cv2.fillConvexPoly(zb, np.stack([u[f[t]], vv[f[t]]], 1).round().astype(np.int32), float(fz[t]))
    # per-pixel depth: refine by barycentric-free approximation (face mean z); adequate at 1-2 mm for a 2.7 cm thick spoon at 0.5 m
    return zb

def rd(p): return cv2.imread(p, cv2.IMREAD_UNCHANGED)

def one(t):
    M = rd(f'{C}/objects/{OBJ}/masks/1/{t:06d}.png') > 0
    hand = (rd(f'{C}/objects/{OBJ}/masks/2/{t:06d}.png') > 0) | (rd(f'{C}/objects/{OBJ}/masks/3/{t:06d}.png') > 0)
    if OTHER: hand |= rd(f'{C}/objects/{OTHER}/masks/1/{t:06d}.png') > 0
    dv = rd(f'{C}/depth/{t:06d}.png').astype(np.float64); D = np.where(dv < 65535, 65535 / np.maximum(dv, 1) - 1, np.nan)
    det = np.array([d['camera_point'] for d in BD[t]]) if BD[t] else np.zeros((0, 3))
    n = len(ROLLS); sil = np.full(n, np.nan); dep = np.full(n, np.nan); off = np.full(n, np.nan); ball = np.full(n, np.nan)
    shx = np.zeros(n); shy = np.zeros(n)
    ys, xs = np.nonzero(M)
    if len(xs) < 300: return t, int(M.sum()), sil, dep, off, ball, shx, shy
    y0, y1, x0, x1 = max(ys.min() - 60, 0), min(ys.max() + 60, H), max(xs.min() - 60, 0), min(xs.max() + 60, W)
    Mc, hc, Dc = M[y0:y1, x0:x1], hand[y0:y1, x0:x1], D[y0:y1, x0:x1]
    for i, r in enumerate(ROLLS):
        T = Tc[t] @ rollmat(r); zb = render(T)[y0:y1, x0:x1]; best = None
        for du in SH:
            for dvv in SH:
                zs = np.roll(np.roll(zb, dvv, 0), du, 1); rm = np.isfinite(zs); occ = hc | (np.isfinite(Dc) & (Dc < zs - 0.015))
                c = ((Mc & ~rm).sum() + (rm & ~Mc & ~occ).sum()) / Mc.sum()
                if best is None or c < best[0]: best = (c, du, dvv, zs)
        c, du, dvv, zs = best; sil[i] = c; shx[i] = du; shy[i] = dvv
        vis = np.isfinite(zs) & Mc & ~hc & np.isfinite(Dc)
        if vis.sum() > 80:
            dz = zs[vis] - Dc[vis]; md = np.median(dz); r_ = np.abs(dz - md); dep[i] = 1000 * np.mean(np.minimum(r_, 0.02)); off[i] = 1000 * md
        if len(det):
            P = BM @ T[:3, :3].T + T[:3, 3]; dd = np.linalg.norm(det[:, None] - P[None], axis=2).min(1)
            inl = dd < 0.12
            if inl.any(): ball[i] = 100 * np.mean(np.minimum(dd[inl], 0.05))
    return t, int(M.sum()), sil, dep, off, ball, shx, shy

if __name__ == '__main__':
    FR = list(range(len(Tc)))
    if os.environ.get('FRAMES'): FR = [int(x) for x in os.environ['FRAMES'].split(',')]
    with Pool(int(os.environ.get('SLURM_CPUS_PER_TASK', 8))) as P: rows = P.map(one, FR)
    rows.sort(key=lambda r: r[0])
    np.savez_compressed(OUT, frames=np.array([r[0] for r in rows]), maskpx=np.array([r[1] for r in rows]), rolls=ROLLS,
                        sil=np.stack([r[2] for r in rows]), dep=np.stack([r[3] for r in rows]), dep_offset=np.stack([r[4] for r in rows]),
                        ball=np.stack([r[5] for r in rows]), shift_u=np.stack([r[6] for r in rows]), shift_v=np.stack([r[7] for r in rows]),
                        long_axis=LONG, com=com, bundle=B)
    print('done', len(rows))
