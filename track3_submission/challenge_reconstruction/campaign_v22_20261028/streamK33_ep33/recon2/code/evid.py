"""K33-recon2 (GT-free) per-frame pose evidence for one object of a bundle: two-sided occlusion-aware silhouette + stereo depth, and the
best small camera-frame correction found by a cheap image-shift search. Occluders = masks/2|3 (hand), the other object's mask, and
pixels whose stereo depth is >1.5 cm in front of the render. Cost(T) = (FN + FP_unoccluded)/|mask|.
For each frame: search pixel shifts (du,dv) at the render depth (= camera x,y translation) and a depth offset from the median
signed depth -> report the camera-frame translation that best explains our images, plus cost at v14 vs best.
usage: evid.py BUNDLE_EP_DIR OBJ OUT.tsv [MESH]"""
import sys, os, json, numpy as np, cv2, trimesh
from multiprocessing import Pool
B, OBJ, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
EPN = int(os.environ.get('EPN', 33)); C = f'/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/canonical/evaluation/episode_{EPN:06d}'
K = json.load(open(f'{C}/intrinsics.json')); fx, fy, cx, cy, W, H = K['fx'], K['fy'], K['cx'], K['cy'], K['width'], K['height']
z = np.load(f'{B}/object_poses.npz', allow_pickle=False); names = list(z['names']); Tc = z['camera_from_object'][:, names.index(OBJ)]
OTHER = ([n for n in names if n != OBJ] or [None])[0]
m = trimesh.load(sys.argv[4] if len(sys.argv) > 4 else f'{B}/meshes/{OBJ}.obj', force='mesh'); v = np.asarray(m.vertices); f = np.asarray(m.faces)
SH = np.arange(-24, 25, 3)
def render(T):
    p = v @ T[:3, :3].T + T[:3, 3]; u = fx * p[:, 0] / p[:, 2] + cx; vv = fy * p[:, 1] / p[:, 2] + cy
    zb = np.full((H, W), np.inf, np.float32); fz = p[f, 2].mean(1)
    for t in np.argsort(-fz):
        cv2.fillConvexPoly(zb, np.stack([u[f[t]], vv[f[t]]], 1).round().astype(np.int32), float(fz[t]))
    return zb
def rd(p): return cv2.imread(p, cv2.IMREAD_UNCHANGED)
def one(t):
    M = rd(f'{C}/objects/{OBJ}/masks/1/{t:06d}.png') > 0
    hand = (rd(f'{C}/objects/{OBJ}/masks/2/{t:06d}.png') > 0) | (rd(f'{C}/objects/{OBJ}/masks/3/{t:06d}.png') > 0) | ((rd(f'{C}/objects/{OTHER}/masks/1/{t:06d}.png') > 0) if OTHER else False)
    dv = rd(f'{C}/depth/{t:06d}.png').astype(np.float64); D = np.where(dv < 65535, 65535 / np.maximum(dv, 1) - 1, np.nan)
    if M.sum() < 50: return [t, int(M.sum())] + [np.nan] * 8
    zb = render(Tc[t]); Zc = float(Tc[t][2, 3])
    best = None
    for du in SH:
        for dvv in SH:
            zs = np.roll(np.roll(zb, dvv, 0), du, 1); rm = np.isfinite(zs)
            occ = hand | (np.isfinite(D) & (D < zs - 0.015))
            c = ((M & ~rm).sum() + (rm & ~M & ~occ).sum()) / M.sum()
            if best is None or c < best[0]: best = (c, du, dvv)
            if du == 0 and dvv == 0: c0 = c
    c, du, dvv = best
    zs = np.roll(np.roll(zb, dvv, 0), du, 1); both = np.isfinite(zs) & M & ~hand & np.isfinite(D)
    sdz = float(np.nanmedian(zs[both] - D[both])) if both.sum() > 50 else np.nan
    both0 = np.isfinite(zb) & M & ~hand & np.isfinite(D); sdz0 = float(np.nanmedian(zb[both0] - D[both0])) if both0.sum() > 50 else np.nan
    dx, dy = du * Zc / fx, dvv * Zc / fy
    return [t, int(M.sum()), c0, c, dx * 100, dy * 100, sdz0 * 100 if sdz0 == sdz0 else np.nan, sdz * 100 if sdz == sdz else np.nan, float(np.isfinite(zb).sum()), Zc * 100]
if __name__ == '__main__':
    with Pool(int(os.environ.get('SLURM_CPUS_PER_TASK', 8))) as P: rows = P.map(one, range(len(Tc)))
    hdr = 'frame\tmaskpx\tcost0\tcostbest\tdx_cm\tdy_cm\tdz0_cm\tdzbest_cm\trenderpx\tZ_cm'
    open(OUT, 'w').write(hdr + '\n' + '\n'.join('\t'.join(f'{x:.3f}' if isinstance(x, float) else str(x) for x in r) for r in rows) + '\n')
    print('done', len(rows))
