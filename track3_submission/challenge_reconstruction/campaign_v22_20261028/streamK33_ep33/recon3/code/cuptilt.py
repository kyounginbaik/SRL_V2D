"""K33-recon3 (GT-free): per-frame cup AXIS scan (the cup is rotationally symmetric; only its axis is scored). Tilts the axis about two
body axes perpendicular to it through the COM on a grid (+-TMAX deg, STEP), scoring the same three-cue cost as spoon360.py (two-sided
occlusion-aware silhouette best over image shifts, stereo-depth residual of visible cup pixels, mean signed depth offset).
usage: cuptilt.py BUNDLE_EP_DIR OUT.npz [TMAX=24] [STEP=6] [FSTEP=2]"""
import sys, os, json, numpy as np, cv2, trimesh
from multiprocessing import Pool
from scipy.spatial.transform import Rotation as Rot
B, OUT = sys.argv[1], sys.argv[2]; TMAX = float(sys.argv[3]) if len(sys.argv) > 3 else 24; STEP = float(sys.argv[4]) if len(sys.argv) > 4 else 6
FSTEP = int(sys.argv[5]) if len(sys.argv) > 5 else 2
C = '/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/canonical/evaluation/episode_000033'
K = json.load(open(f'{C}/intrinsics.json')); fx, fy, cx, cy, W, H = K['fx'], K['fy'], K['cx'], K['cy'], K['width'], K['height']
OBJ, OTHER = 'blue_cup', 'wooden_spoon'
z = np.load(f'{B}/object_poses.npz', allow_pickle=False); names = list(z['names']); Tc = z['camera_from_object'][:, names.index(OBJ)]
m = trimesh.load(f'{B}/meshes/{OBJ}.obj', force='mesh'); v = np.asarray(m.vertices); f = np.asarray(m.faces); com = np.asarray(m.center_mass)
c0 = v - v.mean(0); w, U = np.linalg.eigh(c0.T @ c0 / len(c0)); AX = U[:, 2]; P1 = U[:, 0]; P2 = U[:, 1]
G = np.arange(-TMAX, TMAX + 0.1, STEP); SH = np.arange(-15, 16, 3)
def tilt(a, b):
    E = (Rot.from_rotvec(P1 * np.radians(a)) * Rot.from_rotvec(P2 * np.radians(b))).as_matrix()
    Bm = np.eye(4); Bm[:3, :3] = E; Bm[:3, 3] = com - E @ com; return Bm
def render(T):
    p = v @ T[:3, :3].T + T[:3, 3]; u = fx * p[:, 0] / p[:, 2] + cx; vv = fy * p[:, 1] / p[:, 2] + cy
    zb = np.full((H, W), np.inf, np.float32); fz = p[f, 2].mean(1)
    for t in np.argsort(-fz): cv2.fillConvexPoly(zb, np.stack([u[f[t]], vv[f[t]]], 1).round().astype(np.int32), float(fz[t]))
    return zb
def rd(p): return cv2.imread(p, cv2.IMREAD_UNCHANGED)
def one(t):
    M = rd(f'{C}/objects/{OBJ}/masks/1/{t:06d}.png') > 0
    hand = (rd(f'{C}/objects/{OBJ}/masks/2/{t:06d}.png') > 0) | (rd(f'{C}/objects/{OBJ}/masks/3/{t:06d}.png') > 0) | (rd(f'{C}/objects/{OTHER}/masks/1/{t:06d}.png') > 0)
    dv = rd(f'{C}/depth/{t:06d}.png').astype(np.float64); D = np.where(dv < 65535, 65535 / np.maximum(dv, 1) - 1, np.nan)
    n = len(G); sil = np.full((n, n), np.nan); dep = np.full((n, n), np.nan); off = np.full((n, n), np.nan)
    ys, xs = np.nonzero(M)
    if len(xs) < 300: return t, int(M.sum()), sil, dep, off
    y0, y1, x0, x1 = max(ys.min() - 60, 0), min(ys.max() + 60, H), max(xs.min() - 60, 0), min(xs.max() + 60, W)
    Mc, hc, Dc = M[y0:y1, x0:x1], hand[y0:y1, x0:x1], D[y0:y1, x0:x1]
    for i, a in enumerate(G):
        for j, b in enumerate(G):
            zb = render(Tc[t] @ tilt(a, b))[y0:y1, x0:x1]; best = None
            for du in SH:
                for dvv in SH:
                    zs = np.roll(np.roll(zb, dvv, 0), du, 1); rm = np.isfinite(zs); occ = hc | (np.isfinite(Dc) & (Dc < zs - 0.015))
                    c = ((Mc & ~rm).sum() + (rm & ~Mc & ~occ).sum()) / Mc.sum()
                    if best is None or c < best[0]: best = (c, zs)
            c, zs = best; sil[i, j] = c; vis = np.isfinite(zs) & Mc & ~hc & np.isfinite(Dc)
            if vis.sum() > 80:
                dz = zs[vis] - Dc[vis]; md = np.median(dz); dep[i, j] = 1000 * np.mean(np.minimum(np.abs(dz - md), 0.02)); off[i, j] = 1000 * md
    return t, int(M.sum()), sil, dep, off
if __name__ == '__main__':
    FR = list(range(0, len(Tc), FSTEP))
    with Pool(int(os.environ.get('SLURM_CPUS_PER_TASK', 8))) as P: rows = P.map(one, FR)
    rows.sort(key=lambda r: r[0])
    np.savez_compressed(OUT, frames=np.array([r[0] for r in rows]), maskpx=np.array([r[1] for r in rows]), grid=G, sil=np.stack([r[2] for r in rows]),
                        dep=np.stack([r[3] for r in rows]), dep_offset=np.stack([r[4] for r in rows]), axis=AX, p1=P1, p2=P2, com=com, bundle=B)
    print('done', len(rows))
