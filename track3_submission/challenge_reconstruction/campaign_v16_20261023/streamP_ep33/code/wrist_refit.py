"""Stream P ep33: windowed wrist ILC folded into the phase weights of an existing registered actor (2026-10-24).

The actor architecture is unchanged (same ComponentPolicy class, same keys, same feedback weights, margins and release
primitives). From one physical rollout of the base actor the smoothed error of a held object vs the reference
(rotation R_ach R_ref^T and position p_ach - p_ref, setup frame) is measured in a time window; the commanded wrist pose of the
holding hand is corrected by C = exp(-g_rot * err_rot) about the achieved object centre plus -g_pos * err_pos (smoothstep
ramps), converted to wrist joint commands (world slides + intrinsic XYZ rotation joints), and the per-substep command change
is least-squares projected onto the actor's own radial phase basis and added to phase_weights. The new checkpoint is then
evaluated physically with the unchanged evaluate_policy_component (same frozen scene / episode dir as the base, unless
--episode-dir is given). Only time, measured proprioception enter the actor at run time.
usage: wrist_refit.py BASE_DIR OUT_DIR --side left --object blue_cup --t-on 4.8 --t-off 16.0 --ramp 0.4 --gain-rot 0.8 --gain-pos 0.0
       [--sigma-s 0.4] [--rot-axes xyz] [--episode-dir DIR] [--reference POSES.npz] [--no-eval]"""
import os, sys, json, hashlib, argparse
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(k, '1')
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R
B = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017'); sys.path[:0] = [str(B / 'code'), '/data1/SRL_V2D/third_party/DexAgent']
from wrist_reference import continuous_euler_xyz
from learned_feedback_policy import features
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
ap = argparse.ArgumentParser(); ap.add_argument('base'); ap.add_argument('out'); ap.add_argument('--side', required=True); ap.add_argument('--object', required=True)
ap.add_argument('--t-on', type=float, required=True); ap.add_argument('--t-off', type=float, required=True); ap.add_argument('--ramp', type=float, default=0.4)
ap.add_argument('--gain-rot', type=float, default=0.0); ap.add_argument('--gain-pos', type=float, default=0.0); ap.add_argument('--sigma-s', type=float, default=0.4)
ap.add_argument('--rot-axes', default='xyz'); ap.add_argument('--episode-dir'); ap.add_argument('--reference'); ap.add_argument('--ridge', type=float, default=1e-3); ap.add_argument('--no-eval', action='store_true')
a = ap.parse_args(); base, out = Path(a.base).resolve(), Path(a.out).resolve()
if (out / 'training/policy.npz').exists(): raise SystemExit('exists ' + str(out))
ev = json.loads((base / 'evaluation/result.json').read_text()); scene = ev['execution']['scene']; epd = a.episode_dir or ev['input_dir']
man = json.loads((Path(epd) / 'input_manifest.json').read_text()); S = np.asarray(man['setup_from_source'])
refp = a.reference or str(Path(man['source']) / 'object_poses.npz'); z = np.load(refp); names = [str(x) for x in z['names']]; P = S @ z['world_from_object'][:, names.index(a.object)]
tr = np.load(ev['execution']['trajectory']); oi = [str(x) for x in tr['object_names']].index(a.object); n = min(len(P), len(tr['object_positions'])); fps = 20.0
Ra = R.from_quat(tr['object_quaternions_wxyz'][:n, oi][:, [1, 2, 3, 0]]); pa = tr['object_positions'][:n, oi]
erot = (Ra * R.from_matrix(P[:n, :3, :3]).inv()).as_rotvec(); epos = pa - P[:n, :3, 3]
mask = np.array(['xyz'.index(c) for c in a.rot_axes]); keep = np.zeros(3); keep[mask] = 1; erot = erot * keep
sig = a.sigma_s * fps; erot_s = gaussian_filter1d(erot, sig, axis=0, mode='nearest'); epos_s = gaussian_filter1d(epos, sig, axis=0, mode='nearest')
sm = lambda u: np.clip(u, 0, 1) ** 3 * (10 - 15 * np.clip(u, 0, 1) + 6 * np.clip(u, 0, 1) ** 2)
ft = np.arange(n) / fps; gate = sm((ft - a.t_on) / a.ramp) * (1 - sm((ft - (a.t_off - a.ramp)) / a.ramp))
st = tr['substep_time_s']; sc = tr['substep_ctrl']
with np.load(base / 'training/policy.npz', allow_pickle=False) as zz: d = {k: zz[k] for k in zz.files}
names_act = [str(x) for x in d['actuator_names']]; pre = f'robot_{a.side}_{a.side}_'
ids = [names_act.index(pre + k + '_position') for k in ('pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z')]
g = np.interp(st, ft, gate); er = np.stack([np.interp(st, ft, erot_s[:, i]) for i in range(3)], 1); ep = np.stack([np.interp(st, ft, epos_s[:, i]) for i in range(3)], 1)
c = np.stack([np.interp(st, ft, pa[:, i]) for i in range(3)], 1)
p0 = sc[:, ids[:3]]; ang0 = sc[:, ids[3:]]; C = R.from_rotvec(-a.gain_rot * g[:, None] * er)
p1 = c + C.apply(p0 - c) - a.gain_pos * g[:, None] * ep
R1 = C * R.from_euler('XYZ', ang0); ang1 = np.array([continuous_euler_xyz(R.from_matrix(R1[k].as_matrix()[None]), ang0[k])[0] for k in range(len(st))])
delta = np.concatenate([p1 - p0, ang1 - ang0], 1); delta[g == 0] = 0
phase = st / float(d['duration_s']); Phi = features(phase, d['centers'], float(d['width']))
target = delta / d['action_scale'][ids]
active = np.nonzero((Phi[g > 0].max(0) > 1e-6))[0] if (g > 0).any() else np.array([], int)
A = Phi[:, active]; D = np.linalg.solve(A.T @ A + a.ridge * np.eye(len(active)), A.T @ target)
W = d['phase_weights'].copy(); W[np.ix_(active, ids)] += D; d['phase_weights'] = W
fit = A @ D * d['action_scale'][ids]; fit_err = float(np.abs(fit - delta).max())
(out / 'training').mkdir(parents=True, exist_ok=True); ck = out / 'training/policy.npz'; np.savez_compressed(ck, **d)
rep = json.loads((base / 'training/training_report.json').read_text())
rep.update(checkpoint_sha256=sha(ck), wrist_refit=dict(base_actor=str(base / 'training/policy.npz'), base_sha256=sha(base / 'training/policy.npz'), base_rollout=ev['execution']['trajectory'],
    reference=refp, reference_sha256=sha(refp), side=a.side, object=a.object, t_on=a.t_on, t_off=a.t_off, ramp=a.ramp, gain_rot=a.gain_rot, gain_pos=a.gain_pos, sigma_s=a.sigma_s,
    rot_axes=a.rot_axes, ridge=a.ridge, phase_centers_changed=int(len(active)), max_command_change=dict(pos_m=float(np.abs(delta[:, :3]).max()), rot_rad=float(np.abs(delta[:, 3:]).max())),
    projection_max_error=fit_err, mean_err_in_window=dict(rot_deg=float(np.degrees(np.linalg.norm(erot_s[gate > .5], axis=1).mean())) if (gate > .5).any() else 0., pos_cm=float(np.linalg.norm(epos_s[gate > .5], axis=1).mean() * 100) if (gate > .5).any() else 0.),
    script=str(Path(__file__).resolve()), script_sha256=sha(__file__)))
(out / 'training/training_report.json').write_text(json.dumps(rep, indent=2, default=str))
print('refit', json.dumps(rep['wrist_refit']['max_command_change']), 'proj err', fit_err, 'window err', rep['wrist_refit']['mean_err_in_window'], flush=True)
if a.no_eval: sys.exit()
from evaluate_policy_component import evaluate
evaluate(33, scene, str(ck), str(out / 'evaluation'), epd)
r = json.loads((out / 'evaluation/result.json').read_text()); m = r['metrics']['official_style_reference_proxy']
print('DONE', out.name, json.dumps(dict(mp_sr=m['mp_sr'], objects={k: (round(v['mean_position_error_cm'][0], 3), round(v['mean_orientation_error_degrees'][0], 2), round(v['max_position_error_m'][0] * 100, 1)) for k, v in m['objects'].items()})), flush=True)
