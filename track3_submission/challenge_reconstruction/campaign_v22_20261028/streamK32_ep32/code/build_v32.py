"""K32 A3 (GT-free): K32_jisu2 variants of Jisu's ep32 fix. Spoon only (cup = Jisu). Optional (a) undo Jisu's x1.0255 camera-ray depth scaling
of the spoon (our stereo: Jisu's resting spoon is 11 mm INTO the local stereo table plane, v14's sits on it at +0.2 mm), (b) roll offsets about the
spoon's long PCA axis through the COM on frame segments, smoothstep ramps of RAMP frames. World = world_from_rgb(v14) @ camera_from_object.
usage: build_v32.py LABEL DEPTH(0|1) [a:b:deg ...]"""
import sys, json, shutil, hashlib, numpy as np, trimesh
from pathlib import Path
from scipy.spatial.transform import Rotation as R
K = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK32_ep32'); RAMP = 4
JP = Path('/data1/SRL_V2D/challenge_submissions_SRL/cowork_ep32_kinematic_fixed_20261102/ep32_kinematic_fixed/ep32_fixed_object_poses.npz')
B14 = Path('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/selected20_final_v14_g17/episode_000032')
T14 = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/target_final_v14_g17/episode_000032')
lab, depth = sys.argv[1], int(sys.argv[2]); segs = [tuple(map(float, s.split(':'))) for s in sys.argv[3:]]
J = np.load(JP, allow_pickle=False); b = np.load(B14 / 'object_poses.npz', allow_pickle=False); names = [str(x) for x in J['names']]; k = names.index('wooden_spoon')
cam = J['camera_from_object'].copy(); N = len(cam)
m = trimesh.load(B14 / 'meshes/wooden_spoon.obj', force='mesh'); v = np.asarray(m.vertices); com = np.asarray(m.center_mass)
c0 = v - v.mean(0); w, U = np.linalg.eigh(c0.T @ c0); LONG = U[:, 2] * np.sign(U[np.argmax(np.abs(U[:, 2])), 2])
if depth: cam[:, k, :3, 3] /= 1.0255
sm = lambda u: np.clip(u, 0, 1) ** 3 * (10 - 15 * np.clip(u, 0, 1) + 6 * np.clip(u, 0, 1) ** 2)
off = np.zeros(N)
for a, bb, deg in segs:
    t = np.arange(N); wgt = sm((t - (a - RAMP)) / RAMP) * (1 - sm((t - bb) / RAMP)); off += deg * wgt
for t in np.flatnonzero(off != 0):
    E = R.from_rotvec(LONG * np.radians(off[t])).as_matrix(); Bm = np.eye(4); Bm[:3, :3] = E; Bm[:3, 3] = com - E @ com; cam[t, k] = cam[t, k] @ Bm
W = b['world_from_rgb'][:, None] @ cam
tg = K / 'targets' / lab / 'episode_000032'; tg.mkdir(parents=True)
for i, n in enumerate(names):
    np.savez_compressed(tg / f'{n}_trajectory.npz', T_world_obj=W[:, i].astype(np.float64)); shutil.copy2(T14 / f'{n}_mesh_metric.obj', tg / f'{n}_mesh_metric.obj')
(tg / 'K32_TARGET.json').write_text(json.dumps(dict(stream='K32', label=lab, base='K32_jisu (Jisu ep32 fix, v14 world)', undo_spoon_depth_scale_1_0255=bool(depth),
    roll_segments_deg=segs, ramp_frames=RAMP, long_axis_body=LONG.tolist(), com_body=com.tolist(), ground_truth_used=False), indent=1))
print(lab, 'depth', depth, 'segs', segs, 'frames rolled', int((off != 0).sum()))
