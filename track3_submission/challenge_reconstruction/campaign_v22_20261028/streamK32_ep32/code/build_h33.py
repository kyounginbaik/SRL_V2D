"""K33 hypothesis bundle for ep33 (copy of K35 build_h.py; spoon slot looked up by name, cup untouched) (GT-free). Copies selected20_final_v14_g17/episode_000033 and applies a body-frame roll about the
spoon's long PCA axis through its centre of mass on a frame window, with a smoothstep ramp back to the tracked track:
  corr(t) = c                         for t <= t_s
          = c * (1 - smooth((t-t_s)/(t_e-t_s)))  for t_s < t < t_e,  0 after.
c = 'rest' -> median over the static window [0, t_s] of the per-frame physical rest roll (lowest COM of our rigid mesh dropped onto our
stereo support plane, roll scan +-60 deg, 1 deg), i.e. the roll our mesh must have to lie stably on the measured table.
usage: build_h.py LABEL c|rest t_s t_e  ->  bundles/LABEL/episode_000033 + targets/LABEL/episode_000033"""
import sys, json, shutil, hashlib, numpy as np, trimesh
from pathlib import Path
from scipy.spatial.transform import Rotation as Rot
K = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK32_ep32')
SRC = Path('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/selected20_final_v14_g17/episode_000033')
CAN = Path('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/canonical/evaluation/episode_000033')
lab, cs, ts, te = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
out = K / 'bundles' / lab / 'episode_000033'; assert not out.exists(); shutil.copytree(SRC, out)
z = np.load(SRC / 'object_poses.npz', allow_pickle=False); d = {k: z[k] for k in z.files}; KS = list(d['names']).index('wooden_spoon')
m = trimesh.load(out / 'meshes' / 'wooden_spoon.obj', force='mesh'); v = np.asarray(m.vertices); com = np.asarray(m.center_mass)
c0 = v - v.mean(0); w, U = np.linalg.eigh(c0.T @ c0 / len(c0)); long = U[:, 2]
if long[np.argmax(np.abs(long))] < 0: long = -long
G = np.load(CAN / 'support_plane.npz', allow_pickle=False)['gravity_from_canonical']
def rest_roll(t):
    P = G @ d['world_from_object'][t, KS]; cg = P[:3, :3] @ com + P[:3, 3]; best = None
    for r in np.arange(-60, 61, 1.0):
        E = Rot.from_rotvec(long * np.radians(r)).as_matrix(); vg = ((v - com) @ E.T + com) @ P[:3, :3].T + P[:3, 3]
        h = cg[2] - vg[:, 2].min(); best = min(best or (h, r), (h, r))
    return best[1]
per = [rest_roll(t) for t in range(ts + 1)] if cs == 'rest' else []
c = float(np.median(per)) if cs == 'rest' else float(cs)
sm = lambda u: np.clip(u, 0, 1) ** 3 * (10 - 15 * np.clip(u, 0, 1) + 6 * np.clip(u, 0, 1) ** 2)
corr = np.array([c if t <= ts else (c * (1 - sm((t - ts) / (te - ts))) if t < te else 0.0) for t in range(len(d['world_from_object']))])
for t in np.flatnonzero(corr != 0):
    E = Rot.from_rotvec(long * np.radians(corr[t])).as_matrix(); B = np.eye(4); B[:3, :3] = E; B[:3, 3] = com - E @ com
    d['world_from_object'][t, KS] = d['world_from_object'][t, KS] @ B
    d['camera_from_object'][t, KS] = d['camera_from_object'][t, KS] @ B
(out / 'object_poses.npz').unlink(); np.savez_compressed(out / 'object_poses.npz', **d)
rec = dict(stream='K33', label=lab, source=str(SRC), ground_truth_used=False, rule='body-frame roll about the spoon long PCA axis through the COM',
           c_deg=c, c_mode=cs, per_frame_rest_roll_deg=per, t_s=ts, t_e=te, long_axis_body=long.tolist(), com_body=com.tolist(),
           support_plane=str(CAN / 'support_plane.npz'), frames_changed=int((corr != 0).sum()),
           object_poses_sha256=hashlib.sha256((out / 'object_poses.npz').read_bytes()).hexdigest())
(out / 'K33_CHANGES.json').write_text(json.dumps(rec, indent=1))
tg = K / 'targets' / lab / 'episode_000033'; tg.mkdir(parents=True)
np.savez_compressed(tg / 'wooden_spoon_trajectory.npz', T_world_obj=d['world_from_object'][:, KS])
V14T = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/target_final_v14_g17/episode_000033')
for fn in ('blue_cup_trajectory.npz', 'blue_cup_mesh_metric.obj'): shutil.copy2(V14T / fn, tg / fn)   # cup unchanged (v14 target files)
shutil.copy2(out / 'meshes' / 'wooden_spoon.obj', tg / 'wooden_spoon_mesh_metric.obj')
print(lab, 'c', c, 'per-frame', per, 'changed', rec['frames_changed'])
