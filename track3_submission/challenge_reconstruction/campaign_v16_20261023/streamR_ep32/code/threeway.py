"""Stream R ep32: three-way per-frame comparison robot rollout / reference / stereo-ICP fit (audit_ep32_v2 JSON), in the source
world. Cup: axis angle (continuous-axial metric) and anchor (surface-centroid) position; spoon: anchor position and full rotation.
usage: threeway.py TRAJ_NPZ AUDIT_V2_JSON [--poses OBJECT_POSES_NPZ] [--step 10]"""
import sys, json, argparse, numpy as np, trimesh
from pathlib import Path
from scipy.spatial.transform import Rotation as R
ap = argparse.ArgumentParser(); ap.add_argument('traj'); ap.add_argument('audit'); ap.add_argument('--poses'); ap.add_argument('--step', type=int, default=10); a = ap.parse_args()
FR = Path('/data1/SRL_V2D/challenge_reconstruction/final_recon/episode_000032')
EPD = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/expanded_batch/repaired_export_inputs/episode_000032')
Sinv = np.asarray(json.load(open(EPD / 'input_manifest.json'))['source_from_setup'])
z = np.load(a.poses or FR / 'object_poses.npz'); W = z['world_from_object']; zn = list(z['names'])
tr = np.load(a.traj); tn = [str(x) for x in tr['object_names']]
anc = {o: trimesh.load(FR / f'assets/{o}.obj', force='mesh', process=False).centroid for o in zn}
au = {r['frame']: r for r in json.load(open(a.audit))['rows']}
def ach(o, k):
    i = tn.index(o); T = np.eye(4); T[:3, :3] = R.from_quat(tr['object_quaternions_wxyz'][k, i][[1, 2, 3, 0]]).as_matrix(); T[:3, 3] = tr['object_positions'][k, i]; return Sinv @ T
P = lambda T, o: T[:3, :3] @ anc[o] + T[:3, 3]
axang = lambda A, B: np.degrees(np.arccos(np.clip(A[:3, 2] @ B[:3, 2], -1, 1)))
rot = lambda A, B: np.degrees(R.from_matrix(A[:3, :3].T @ B[:3, :3]).magnitude())
n = min(len(W), len(tr['qpos'])); rows = []
for k in range(n):
    Ac, Rc = ach('blue_cup', k), W[k, zn.index('blue_cup')]; As, Rs = ach('wooden_spoon', k), W[k, zn.index('wooden_spoon')]
    r = dict(c_ar_deg=axang(Ac, Rc), c_ar_cm=np.linalg.norm(P(Ac, 'blue_cup') - P(Rc, 'blue_cup'))*100, s_ar_cm=np.linalg.norm(P(As, 'wooden_spoon') - P(Rs, 'wooden_spoon'))*100, s_ar_deg=rot(As, Rs))
    q = au.get(k, {})
    if 'cup_fit_T' in q:
        F = np.array(q['cup_fit_T']); r.update(c_af_deg=axang(Ac, F), c_rf_deg=axang(Rc, F), c_af_cm=np.linalg.norm(P(Ac, 'blue_cup') - P(F, 'blue_cup'))*100, c_rf_cm=np.linalg.norm(P(Rc, 'blue_cup') - P(F, 'blue_cup'))*100)
    if 'spoon_fit_T' in q:
        F = np.array(q['spoon_fit_T']); r.update(s_af_cm=np.linalg.norm(P(As, 'wooden_spoon') - P(F, 'wooden_spoon'))*100, s_rf_cm=np.linalg.norm(P(Rs, 'wooden_spoon') - P(F, 'wooden_spoon'))*100, s_af_deg=rot(As, F), s_rf_deg=rot(Rs, F))
    rows.append(r)
keys = ['c_ar_deg', 'c_af_deg', 'c_rf_deg', 'c_ar_cm', 'c_af_cm', 'c_rf_cm', 's_ar_cm', 's_af_cm', 's_rf_cm', 's_ar_deg', 's_af_deg', 's_rf_deg']
print('  t   ' + ' '.join(f'{k:>8}' for k in keys) + '   (a=robot r=reference f=stereo fit)')
for s in list(range(0, n, a.step)) + ['all']:
    g = rows if s == 'all' else rows[s:s+a.step]
    print(('  all ' if s == 'all' else f'{s/20:5.1f} ') + ' '.join(f"{np.nanmean([x.get(k, np.nan) for x in g]):8.2f}" for k in keys))
