"""K32 A1 (GT-free): per-frame image evidence for the ep32 spoon ROLL about its long PCA axis (through the COM), around Jisu's pose.
For roll offsets r (deg) applied to Jisu's camera_from_object: (a) two-sided occlusion-aware silhouette cost (FN + FP_unoccluded)/|mask|
(occluders = hand masks 2|3, cup mask, stereo depth > 1.5 cm in front of the render), (b) median |render depth - stereo depth| on visible
mask pixels (mm; concave/convex side), (c) ball reprojection: median over detections within 12 cm of the spoon of the distance (px) to the
nearest projected ball (ball_models_v1). Also reports the roll offset of v14 and of v26 relative to Jisu per frame.
usage: evroll32.py OUT.tsv f0:f1:step[,..]"""
import sys, os, json, numpy as np, cv2, trimesh
from multiprocessing import Pool
from scipy.spatial.transform import Rotation as R
C = '/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/canonical/evaluation/episode_000032'
K = json.load(open(f'{C}/intrinsics.json')); fx, fy, cx, cy, W, H = K['fx'], K['fy'], K['cx'], K['cy'], K['width'], K['height']
J = np.load('/data1/SRL_V2D/challenge_submissions_SRL/cowork_ep32_kinematic_fixed_20261102/ep32_kinematic_fixed/ep32_fixed_object_poses.npz', allow_pickle=False)
V14 = np.load('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/selected20_final_v14_g17/episode_000032/object_poses.npz', allow_pickle=False)
V26 = np.load('/data1/SRL_V2D/challenge_submissions_SRL/recon_v26_selected20_final_v8_g10b/episode_000032/object_poses.npz', allow_pickle=False)
k = list(J['names']).index('wooden_spoon'); TJ = J['camera_from_object'][:, k]; T14 = V14['camera_from_object'][:, k]; T26 = V26['camera_from_object'][:, k]
m = trimesh.load('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/selected20_final_v14_g17/episode_000032/meshes/wooden_spoon.obj', force='mesh')
v = np.asarray(m.vertices); f = np.asarray(m.faces); com = np.asarray(m.center_mass)
c0 = v - v.mean(0); w, U = np.linalg.eigh(c0.T @ c0); LONG = U[:, 2] * np.sign(U[np.argmax(np.abs(U[:, 2])), 2])
BALLS = np.asarray(json.load(open(f'{C}/ball_models_v1.json'))['wooden_spoon'])
DET = json.load(open(f'{C}/ball_detections_wooden_spoon.json'))['frames']
OFF = np.arange(-180, 180, 10)
def rollT(T, r):
    E = R.from_rotvec(LONG * np.radians(r)).as_matrix(); B = np.eye(4); B[:3, :3] = E; B[:3, 3] = com - E @ com; return T @ B
def roll_of(Ta, Tb):   # body-frame roll of Tb relative to Ta about LONG
    d = R.from_matrix(Ta[:3, :3].T @ Tb[:3, :3]).as_rotvec(); return float(np.degrees(d @ LONG)), float(np.degrees(np.linalg.norm(d - (d @ LONG) * LONG)))
def render(T):
    p = v @ T[:3, :3].T + T[:3, 3]; u = fx * p[:, 0] / p[:, 2] + cx; vv = fy * p[:, 1] / p[:, 2] + cy
    zb = np.full((H, W), np.inf, np.float32); fz = p[f, 2].mean(1)
    for t in np.argsort(-fz): cv2.fillConvexPoly(zb, np.stack([u[f[t]], vv[f[t]]], 1).round().astype(np.int32), float(fz[t]))
    return zb
def rd(p): return cv2.imread(p, cv2.IMREAD_UNCHANGED)
def one(t):
    Mk = rd(f'{C}/objects/wooden_spoon/masks/1/{t:06d}.png') > 0
    hand = (rd(f'{C}/objects/wooden_spoon/masks/2/{t:06d}.png') > 0) | (rd(f'{C}/objects/wooden_spoon/masks/3/{t:06d}.png') > 0) | (rd(f'{C}/objects/blue_cup/masks/1/{t:06d}.png') > 0)
    dv = rd(f'{C}/depth/{t:06d}.png').astype(np.float64); D = np.where(dv < 65535, 65535 / np.maximum(dv, 1) - 1, np.nan)
    det = [d for d in (DET[t] if t < len(DET) else []) if np.linalg.norm(np.asarray(d['camera_point']) - (TJ[t][:3, :3] @ com + TJ[t][:3, 3])) < 0.12]
    r14, t14 = roll_of(TJ[t], T14[t]); r26, t26 = roll_of(TJ[t], T26[t]); rows = []
    for r in OFF:
        T = rollT(TJ[t], r); zb = render(T); rm = np.isfinite(zb)
        occ = hand | (np.isfinite(D) & (D < zb - 0.015))
        sil = ((Mk & ~rm).sum() + (rm & ~Mk & ~occ).sum()) / max(Mk.sum(), 1)
        both = rm & Mk & ~hand & np.isfinite(D); dz = float(np.nanmedian(np.abs(zb[both] - D[both]))) * 1000 if both.sum() > 30 else np.nan
        if det:
            pb = BALLS @ T[:3, :3].T + T[:3, 3]; ub = np.c_[fx * pb[:, 0] / pb[:, 2] + cx, fy * pb[:, 1] / pb[:, 2] + cy]
            bl = float(np.median([min(40.0, np.min(np.linalg.norm(ub - np.asarray(d['uv']), axis=1))) for d in det]))
        else: bl = np.nan
        rows.append((t, int(Mk.sum()), len(det), int(r), sil, dz, bl, r14, r26))
    return rows
if __name__ == '__main__':
    fr = []
    for s in sys.argv[2].split(','):
        a, b, st = map(int, s.split(':')); fr += list(range(a, b, st))
    with Pool(int(os.environ.get('SLURM_CPUS_PER_TASK', 8))) as P: res = P.map(one, fr)
    with open(sys.argv[1], 'w') as o:
        o.write('frame\tmaskpx\tnballs\troll_off\tsil\tdz_mm\tball_px\tv14_roll_off\tv26_roll_off\n')
        for rows in res:
            for r in rows: o.write('\t'.join(f'{x:.3f}' if isinstance(x, float) else str(x) for x in r) + '\n')
    print('done', len(fr))
