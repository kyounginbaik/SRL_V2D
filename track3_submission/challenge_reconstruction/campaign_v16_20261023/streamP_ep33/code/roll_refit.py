"""Stream P ep33: robot re-plan for the corrected spoon roll, folded into the phase weights of an existing registered actor
(same mechanism as wrist_refit.py). The commanded RIGHT-wrist pose is rotated about the spoon's EXECUTED long axis (body x,
through the executed spoon origin, from one physical rollout of the base actor) by gain * phi(t + lead), phi = the reference roll
correction (roll_correction.json, per frame), so the held spoon follows the corrected roll; fingers, left hand, margins and
release primitives unchanged. Command change projected onto the actor's radial phase basis; then a fresh physical evaluation
with the unchanged evaluate_policy_component against the CORRECTED episode dir (and re-scored against the original).
usage: roll_refit.py BASE_DIR OUT_DIR --gain 1.0 [--lead 0.0] [--stretch 1.0] [--ridge 1e-3]"""
import os, sys, json, hashlib, argparse
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(k, '1')
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
B = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017'); sys.path[:0] = [str(B / 'code'), '/data1/SRL_V2D/third_party/DexAgent']
from wrist_reference import continuous_euler_xyz
from learned_feedback_policy import features
P = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamP_ep33')
OEP = P / 'staging/episode_corrected/episode_000033'; REC = P / 'staging/source_corrected/episode_000033/roll_correction.json'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
ap = argparse.ArgumentParser(); ap.add_argument('base'); ap.add_argument('out'); ap.add_argument('--gain', type=float, default=1.0); ap.add_argument('--lead', type=float, default=0.0)
ap.add_argument('--ridge', type=float, default=1e-3); ap.add_argument('--axis', default='spoon', choices=['spoon', 'wristz'], help='spoon: exact roll about the executed spoon long axis (Euler path crosses the XYZ gimbal near pitch -84 deg); wristz: rotate about the commanded wrist body z axis (rot_z joint only, 25-35 deg from the spoon axis), translation keeps the executed spoon origin fixed'); a = ap.parse_args(); base, out = Path(a.base).resolve(), Path(a.out).resolve()
if (out / 'training/policy.npz').exists(): raise SystemExit('exists ' + str(out))
ev = json.loads((base / 'evaluation/result.json').read_text()); scene = ev['execution']['scene']
rec = json.loads(REC.read_text()); fps = 20.0; n = 348; phi_f = np.zeros(n)
for k, v in rec['phi_deg_per_frame'].items(): phi_f[int(k)] = v
tr = np.load(ev['execution']['trajectory']); si = [str(x) for x in tr['object_names']].index('wooden_spoon')
st = tr['substep_time_s']; sc = tr['substep_ctrl']; ft = np.arange(n) / fps
phi = a.gain * np.interp(st + a.lead, ft, phi_f)
ps = np.stack([np.interp(st, ft, tr['object_positions'][:n, si, i]) for i in range(3)], 1)
qf = tr['object_quaternions_wxyz'][:n, si][:, [1, 2, 3, 0]]; Rf = R.from_quat(qf)
from scipy.spatial.transform import Slerp
Rs = Slerp(ft, Rf)(np.clip(st, ft[0], ft[-1]))
with np.load(base / 'training/policy.npz', allow_pickle=False) as zz: d = {k: zz[k] for k in zz.files}
names_act = [str(x) for x in d['actuator_names']]; pre = 'robot_right_right_'
ids = [names_act.index(pre + k + '_position') for k in ('pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z')]
act = np.abs(phi) > 1e-9; p0 = sc[:, ids[:3]]; ang0 = sc[:, ids[3:]]; Rw = R.from_euler('XYZ', ang0)
if a.axis == 'wristz':
    C = Rw * R.from_euler('z', phi[:, None], degrees=True) * Rw.inv(); p1 = ps + C.apply(p0 - ps)
    ang1 = ang0.copy(); ang1[:, 2] += np.radians(phi)
else:
    C = Rs * R.from_euler('x', phi[:, None], degrees=True) * Rs.inv(); p1 = ps + C.apply(p0 - ps); R1 = C * Rw
ang1 = ang1 if a.axis == 'wristz' else ang0.copy(); prev = None
for k in (np.flatnonzero(act) if a.axis == 'spoon' else []):   # continuous branch along time, anchored at the unmodified command
    anchor = ang0[k] if prev is None or k - prev > 1 else ang1[prev] + (ang0[k] - ang0[prev])
    ang1[k] = continuous_euler_xyz(R.from_matrix(R1[k].as_matrix()[None]), anchor)[0]; prev = k
delta = np.concatenate([p1 - p0, ang1 - ang0], 1); delta[~act] = 0
Phi = features(st / float(d['duration_s']), d['centers'], float(d['width'])); target = delta / d['action_scale'][ids]
active = np.nonzero(Phi[act].max(0) > 1e-6)[0]; A = Phi[:, active]; D = np.linalg.solve(A.T @ A + a.ridge * np.eye(len(active)), A.T @ target)
W = d['phase_weights'].copy(); W[np.ix_(active, ids)] += D; d['phase_weights'] = W; fit = A @ D * d['action_scale'][ids]
perr = np.abs(fit - delta).max(0)
(out / 'training').mkdir(parents=True, exist_ok=True); ck = out / 'training/policy.npz'; np.savez_compressed(ck, **d)
rep = json.loads((base / 'training/training_report.json').read_text())
rep.update(checkpoint_sha256=sha(ck), roll_refit=dict(base_actor=str(base / 'training/policy.npz'), base_sha256=sha(base / 'training/policy.npz'), base_rollout=ev['execution']['trajectory'],
    roll_correction=str(REC), roll_correction_sha256=sha(REC), gain=a.gain, lead_s=a.lead, axis=a.axis, ridge=a.ridge, phase_centers_changed=int(len(active)),
    max_command_change=dict(pos_m=float(np.abs(delta[:, :3]).max()), rot_rad=float(np.abs(delta[:, 3:]).max())), projection_max_error=perr.tolist(), script=str(Path(__file__).resolve()), script_sha256=sha(__file__)))
(out / 'training/training_report.json').write_text(json.dumps(rep, indent=2, default=str))
print('refit', rep['roll_refit']['max_command_change'], 'proj err', np.round(perr, 4).tolist(), flush=True)
from evaluate_policy_component import evaluate
evaluate(33, scene, str(ck), str(out / 'evaluation'), str(OEP))
r = json.loads((out / 'evaluation/result.json').read_text()); m = r['metrics']['official_style_reference_proxy']
sys.path.insert(0, str(P / 'code')); os.environ['RESCORE_EPD'] = str(B / 'episodes/episode_000033'); import importlib, rescore; importlib.reload(rescore)
orig = rescore.score(r['execution']['trajectory'], '/data1/SRL_V2D/challenge_reconstruction/final_recon/episode_000033/object_poses.npz')
print('DONE', out.name, json.dumps(dict(mp_sr=m['mp_sr'], corrected={k: (round(v['mean_position_error_cm'][0], 3), round(v['mean_orientation_error_degrees'][0], 2), round(v['max_position_error_m'][0] * 100, 1)) for k, v in m['objects'].items()}, original=orig)), flush=True)
