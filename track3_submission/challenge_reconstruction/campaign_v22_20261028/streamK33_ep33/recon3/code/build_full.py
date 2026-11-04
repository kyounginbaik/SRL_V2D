"""K33-recon3 (GT-free): re-decide the spoon roll about its long PCA axis (through the COM) on EVERY frame by a Viterbi over the full-circle
scan of spoon360.py, then write a hypothesis bundle + scorer target (cup unchanged = v14).
Cues: ball (3D stereo ball detections, cm) is the primary unary; silhouette enters with a small weight. The stereo-depth cue is NOT used:
on the PASSING spoon episodes 3/5/6/36/37/35 with the same G18 mesh it prefers a near-flip (|roll|>=130) on 50-70 % of frames
(out/controls.txt), i.e. it is biased for this mesh, not a defect detector. Pairwise: LAM * |delta roll| (deg/10) between frames.
usage: build_full.py LABEL S360.npz LAM WSIL [smooth_sigma_frames]"""
import sys, json, shutil, hashlib, numpy as np, trimesh
from pathlib import Path
from scipy.spatial.transform import Rotation as Rot
from scipy.ndimage import gaussian_filter1d
K = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK33_ep33')
SRC = Path('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/selected20_final_v14_g17/episode_000033')
lab, scan, LAM, WSIL = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]); SIG = float(sys.argv[5]) if len(sys.argv) > 5 else 2.0
z = np.load(scan, allow_pickle=False); R = z['rolls']; n = len(R); T = len(z['frames'])
ball = np.nan_to_num(z['ball'] - np.nanmin(z['ball'], 1, keepdims=True), nan=0.0)           # cm above the frame's best
sil = np.nan_to_num(z['sil'] - np.nanmin(z['sil'], 1, keepdims=True), nan=0.0)
U = ball + WSIL * sil * 10
dR = np.abs(((R[:, None] - R[None]) + 180) % 360 - 180) / 10.0
cost = U[0].copy(); bp = np.zeros((T, n), int)
for t in range(1, T):
    c = cost[:, None] + LAM * dR; bp[t] = np.argmin(c, 0); cost = c[bp[t], np.arange(n)] + U[t]
path = np.zeros(T, int); path[-1] = int(np.argmin(cost))
for t in range(T - 1, 0, -1): path[t - 1] = bp[t, path[t]]
roll = ((R[path] + 180) % 360) - 180
rr = np.radians(roll); cs, sn = (gaussian_filter1d(np.cos(rr), SIG, mode='nearest'), gaussian_filter1d(np.sin(rr), SIG, mode='nearest')) if SIG > 0 else (np.cos(rr), np.sin(rr))
roll_s = np.degrees(np.arctan2(sn, cs))   # circular smoothing (no unwrap)
out = K / 'bundles' / lab / 'episode_000033'; assert not out.exists(); shutil.copytree(SRC, out)
P = np.load(SRC / 'object_poses.npz', allow_pickle=False); d = {k: P[k] for k in P.files}; KS = list(d['names']).index('wooden_spoon')
LONG, com = z['long_axis'], z['com']
for t in range(T):
    E = Rot.from_rotvec(LONG * np.radians(roll_s[t])).as_matrix(); B = np.eye(4); B[:3, :3] = E; B[:3, 3] = com - E @ com
    d['world_from_object'][t, KS] = d['world_from_object'][t, KS] @ B; d['camera_from_object'][t, KS] = d['camera_from_object'][t, KS] @ B
(out / 'object_poses.npz').unlink(); np.savez_compressed(out / 'object_poses.npz', **d)
rec = dict(stream='K33-recon3', label=lab, source=str(SRC), ground_truth_used=False, scan=scan, lam=LAM, wsil=WSIL, sigma=SIG,
           roll_deg=[round(float(x), 2) for x in roll_s], object_poses_sha256=hashlib.sha256((out / 'object_poses.npz').read_bytes()).hexdigest())
(out / 'K33_CHANGES.json').write_text(json.dumps(rec))
tg = K / 'targets' / lab / 'episode_000033'; tg.mkdir(parents=True)
np.savez_compressed(tg / 'wooden_spoon_trajectory.npz', T_world_obj=d['world_from_object'][:, KS])
V14T = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/target_final_v14_g17/episode_000033')
for fn in ('blue_cup_trajectory.npz', 'blue_cup_mesh_metric.obj'): shutil.copy2(V14T / fn, tg / fn)
shutil.copy2(out / 'meshes' / 'wooden_spoon.obj', tg / 'wooden_spoon_mesh_metric.obj')
for a in range(0, T, 20): print(f'f{a:3d}-{min(a+20,T)-1:3d} roll mean {roll_s[a:a+20].mean():7.1f} min {roll_s[a:a+20].min():7.1f} max {roll_s[a:a+20].max():7.1f}')
print(lab, 'mean |roll|', np.abs(roll_s).mean())
