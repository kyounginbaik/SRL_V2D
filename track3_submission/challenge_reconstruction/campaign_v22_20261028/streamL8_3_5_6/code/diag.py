"""L8 diagnosis: per-frame scorer-faithful errors of an actor rollout (evaluation/execution/trajectory.npz) vs target_final_v5,
with the spoon orientation error split into roll about the spoon long axis vs tilt of the long axis, plus hand-spoon contact force.
usage: diag.py EP ACTOR_DIR MESH_DIR [--every 10]"""
import sys, json, argparse
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, str(Path(__file__).parent)); import plib as B
ap = argparse.ArgumentParser(); ap.add_argument('ep', type=int); ap.add_argument('actor'); ap.add_argument('meshdir'); ap.add_argument('--every', type=int, default=10)
ap.add_argument('--traj'); ap.add_argument('--save')
a = ap.parse_args()
tr = np.load(a.traj or Path(a.actor) / 'evaluation/execution/trajectory.npz')
q = tr['qpos']; names = [str(s) for s in tr['joint_names']]
# object free joints: first 7 qpos of each object (positions in trajectory object_positions)
P = {}
on = [str(s) for s in tr['object_names']]
for i, n in enumerate(on):
    T = np.tile(np.eye(4), (len(q), 1, 1)); T[:, :3, :3] = R.from_quat(tr['object_quaternions_wxyz'][:, i], scalar_first=True).as_matrix(); T[:, :3, 3] = tr['object_positions'][:, i]; P[n] = T
reg = B.register(a.ep, {n: Path(a.meshdir) / f'{n}.obj' for n in B.NAMES})
e, ach, ref = B.per_frame_full(a.ep, P, reg)
print('MEAN', {n: (round(float(e[n][0].mean()), 3), round(float(e[n][1].mean()), 2)) for n in B.NAMES}, 'MPPE', round(float(np.mean([e[n][0] for n in B.NAMES])), 3))
# spoon long axis in the registered (target) body frame: principal axis of the target mesh
g = B.geometry(a.ep)['wooden_spoon']; v = g['v'] - g['v'].mean(0); w, U = np.linalg.eigh(v.T @ v); ax = U[:, -1]
s = B.NAMES.index('wooden_spoon')
Ra = R.from_quat(ach[:, s, 3:]); Rr = R.from_quat(ref[:, s, 3:])
D = (Rr.inv() * Ra).as_rotvec()          # body-frame error rotvec
ang = np.linalg.norm(D, axis=1)
# swing-twist: twist about ax
qd = (Rr.inv() * Ra).as_quat(); tw = 2 * np.degrees(np.arctan2(qd[:, :3] @ ax, qd[:, 3])); tw = (tw + 180) % 360 - 180
axa = Ra.apply(ax); axr = Rr.apply(ax); tilt = np.degrees(np.arccos(np.clip(np.abs((axa * axr).sum(1)), -1, 1)))
f = tr['hand_object_force_n']
print(f'{"t":>5} | pot cm/deg | spoon cm/deg | roll(twist) | axis-tilt | F(on0,on1)  objects={on}')
for k in range(0, len(q), a.every):
    sl = slice(k, k + a.every)
    print(f'{k/20:5.1f} | {e["white_pot"][0][sl].mean():4.2f}/{e["white_pot"][1][sl].mean():5.1f} | {e["wooden_spoon"][0][sl].mean():4.2f}/{e["wooden_spoon"][1][sl].mean():5.1f} | {tw[sl].mean():7.1f} | {tilt[sl].mean():6.1f} | {f[sl,0].mean():6.1f} {f[sl,1].mean():6.1f}')
if a.save: np.savez(a.save, pot=np.c_[e['white_pot']], spoon=np.c_[e['wooden_spoon']], twist=tw, tilt=tilt, force=f)
