"""A20-inhand (GT-free): per-frame best spoon roll offset (about the long PCA axis through the COM, build_h33 convention) from OUR 3D stereo
ball detections. Inliers = detections within 12 cm of the bundle spoon COM (others are far-background/hand false positives, ~1 m away).
For each frame with >=2 inliers: scan roll -180..179 (1 deg), residual = mean nearest distance det->model ball; report best roll,
residual at 0 and at best, and the 2nd-best basin (to show ambiguity). usage: ballroll33.py BUNDLE_EP_DIR out.txt"""
import sys, json, numpy as np, trimesh
from scipy.spatial.transform import Rotation as Rot
B = sys.argv[1]
C = '/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/canonical/evaluation/episode_000033'
z = np.load(f'{B}/object_poses.npz', allow_pickle=False); S = list(z['names']).index('wooden_spoon'); Tc = z['camera_from_object'][:, S]
m = trimesh.load(f'{B}/meshes/wooden_spoon.obj', force='mesh'); v = np.asarray(m.vertices); com = np.asarray(m.center_mass)
c0 = v - v.mean(0); w, U = np.linalg.eigh(c0.T @ c0 / len(c0)); long = U[:, 2]
if long[np.argmax(np.abs(long))] < 0: long = -long
bc = json.load(open('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/streamG18_spoonmesh/out/ballcenters.json'))
pe = bc['per_episode_centres']; M = np.array([np.median([pe[e][k] for e in pe], 0) for k in ('near', 'far', 'tip')])
fr = json.load(open(f'{C}/ball_detections_wooden_spoon.json'))['frames']
rolls = np.arange(-180, 180, 1.0)
Ms = []
for r in rolls:
    E = Rot.from_rotvec(long * np.radians(r)).as_matrix(); Ms.append((M - com) @ E.T + com)
Ms = np.array(Ms)
lines = ['frame ninl  res0_cm  best_roll res_best_cm   alt_basin(roll,res)  inlier_dists_to_v14']
for t in range(len(Tc)):
    D = np.array([d['camera_point'] for d in fr[t]]) if fr[t] else np.zeros((0, 3))
    cc = Tc[t][:3, :3] @ com + Tc[t][:3, 3]
    D = D[np.linalg.norm(D - cc, axis=1) < 0.12] if len(D) else D
    if len(D) < 2: continue
    P = Ms @ Tc[t][:3, :3].T + Tc[t][:3, 3]                       # (R,3,3)
    res = np.linalg.norm(D[None, :, None] - P[:, None], axis=3).min(2).mean(1) * 100
    i0 = np.where(rolls == 0)[0][0]; ib = int(np.argmin(res))
    far = np.abs(((rolls - rolls[ib] + 180) % 360) - 180) > 60
    ia = np.where(far)[0][np.argmin(res[far])]
    lines.append(f'f{t:3d} {len(D)}  {res[i0]:5.2f}  {rolls[ib]:6.0f}  {res[ib]:5.2f}   ({rolls[ia]:5.0f},{res[ia]:5.2f})')
open(sys.argv[2], 'w').write('\n'.join(lines) + '\n'); print('\n'.join(lines))
