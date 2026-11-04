"""Stream R ep32: per-frame comparison of an executed rollout (setup frame) with the reference and with the stereo-ICP fit poses
(audit JSON). Reports, per 0.5 s bin: cup axis tilt (setup z) for rollout / reference / stereo fit, cup bottom z, spoon pose
error vs reference and vs stereo fit, and the official-style score of the rollout against a given object_poses.npz.
usage: compare_rollout.py TRAJ_NPZ AUDIT_JSON [--poses OBJECT_POSES_NPZ]"""
import sys, json, argparse, numpy as np, trimesh
from pathlib import Path
from scipy.spatial.transform import Rotation as R
sys.path[:0] = ['/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code']
import evaluate_mp_sr as E
ap = argparse.ArgumentParser(); ap.add_argument('traj'); ap.add_argument('audit'); ap.add_argument('--poses'); a = ap.parse_args()
FR = Path('/data1/SRL_V2D/challenge_reconstruction/final_recon/episode_000032')
EPD = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/expanded_batch/repaired_export_inputs/episode_000032')
S = np.asarray(json.load(open(EPD / 'input_manifest.json'))['setup_from_source']); Sinv = np.linalg.inv(S)
poses = Path(a.poses) if a.poses else FR / 'object_poses.npz'
tr = np.load(a.traj); names = [str(x) for x in tr['object_names']]
z = np.load(poses); W = z['world_from_object']; zn = list(z['names'])
cv = np.asarray(trimesh.load(FR / 'assets/blue_cup.obj', force='mesh').vertices)
n = min(len(W), len(tr['qpos']))
def T_of(i, k):
    T = np.eye(4); T[:3, :3] = R.from_quat(tr['object_quaternions_wxyz'][k, i][[1, 2, 3, 0]]).as_matrix(); T[:3, 3] = tr['object_positions'][k, i]; return T
au = {r['frame']: r for r in json.load(open(a.audit))['rows']}
rows = []
for k in range(n):
    Ta = T_of(names.index('blue_cup'), k); Tr = S @ W[k, zn.index('blue_cup')]
    r = dict(t=k/20, ach_tilt=np.degrees(np.arccos(Ta[2, 2])), ref_tilt=np.degrees(np.arccos(Tr[2, 2])), ach_bot=(cv @ Ta[:3, :3].T + Ta[:3, 3])[:, 2].min()*100,
             ref_bot=(cv @ Tr[:3, :3].T + Tr[:3, 3])[:, 2].min()*100, axis_err=np.degrees(np.arccos(np.clip(Ta[:3, 2] @ Tr[:3, 2], -1, 1))))
    if k in au and 'fit_cup_tilt_setup_deg' in au[k]: r.update(fit_tilt=au[k]['fit_cup_tilt_setup_deg'], fit_bot=au[k]['fit_cup_bottom_z_cm'])
    rows.append(r)
print('  t   | cup tilt ach ref fit | bottom z ach ref fit | axis err ach-ref')
for s in range(0, n, 10):
    g = rows[s:s+10]; f = lambda key: np.nanmean([x.get(key, np.nan) for x in g])
    print(f"{s/20:5.1f} | {f('ach_tilt'):5.1f} {f('ref_tilt'):5.1f} {f('fit_tilt'):5.1f} | {f('ach_bot'):5.1f} {f('ref_bot'):5.1f} {f('fit_bot'):5.1f} | {f('axis_err'):5.1f}")
ref = E.load_reference(poses.parent) if poses.name == 'object_poses.npz' else None
