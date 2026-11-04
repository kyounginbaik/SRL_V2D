"""Stream P ep33 (phase 5b): grasp-aware robot roll re-plan folded into the phase weights of an existing registered actor
(same mechanism as wrist_refit.py / roll_refit.py; architecture, feedback weights, margins and release primitives unchanged).
Per substep of one physical rollout of the base actor:
  - right wrist: commanded pose rotated about the EXECUTED spoon long axis (body x through the executed spoon origin) by a robot
    roll profile phi_r(t) (PCHIP through --profile knots 'frame:deg,...', or gain * reference correction shifted by --lead), with
    --axis spoon (exact, branch-continuous Euler) or --axis wristz (rot_z joint only, origin held);
  - fingers (optional): time-windowed closing bias added to the commands: --squeeze S rad on the thumb-index pinch joints
    (thumb CMC_FE, MCP_FE, IP; index PIP, DIP) and --close3 C rad on middle/ring/pinky MCP_FE/PIP/DIP (wrap the handle),
    smoothstep in/out over --sq-ramp s inside [--sq-t0, --sq-t1].
The command change is least-squares projected onto the actor's radial phase basis; then a fresh physical evaluation with the
unchanged evaluate_policy_component against the roll-CORRECTED episode dir, re-scored against the original, plus window diagnostics.
usage: roll_refit2.py BASE_DIR OUT_DIR [--profile K] [--gain G --lead L] [--axis spoon|wristz] [--squeeze S] [--close3 C] [--sq-t0 T --sq-t1 T --sq-ramp R]"""
import os, sys, json, hashlib, argparse
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(k, '1')
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation as R, Slerp
B = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017'); sys.path[:0] = [str(B / 'code'), '/data1/SRL_V2D/third_party/DexAgent']
from wrist_reference import continuous_euler_xyz
from learned_feedback_policy import features
P = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamP_ep33')
OEP = P / 'staging/episode_corrected/episode_000033'; REC = P / 'staging/source_corrected/episode_000033/roll_correction.json'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
ap = argparse.ArgumentParser(); ap.add_argument('base'); ap.add_argument('out'); ap.add_argument('--profile'); ap.add_argument('--gain', type=float, default=0.0); ap.add_argument('--lead', type=float, default=0.0)
ap.add_argument('--axis', default='spoon', choices=['spoon', 'wristz']); ap.add_argument('--squeeze', type=float, default=0.0); ap.add_argument('--close3', type=float, default=0.0)
ap.add_argument('--sq-t0', type=float, default=6.3); ap.add_argument('--sq-t1', type=float, default=8.6); ap.add_argument('--sq-ramp', type=float, default=0.3); ap.add_argument('--ridge', type=float, default=1e-3)
ap.add_argument('--offset', default='0,0,0', help='world (setup) translation of the right wrist in cm, applied in [--off-t0, --off-t1] with --off-ramp smoothstep ramps'); ap.add_argument('--off-t0', type=float, default=6.6); ap.add_argument('--off-t1', type=float, default=7.6); ap.add_argument('--off-ramp', type=float, default=0.25)
ap.add_argument('--no-eval', action='store_true'); ap.add_argument('--hold-end', action='store_true', help='keep the last profile value to the end of the episode (e.g. 360 deg: same physical orientation, wrist joints stay wound)')
ap.add_argument('--fb-comp', action='store_true', help='divide out the actor\'s diagonal proprioceptive feedback (command change = delta * (1 - F_jj action_scale_j / state_scale_j))')
a = ap.parse_args(); base, out = Path(a.base).resolve(), Path(a.out).resolve()
if (out / 'training/policy.npz').exists(): raise SystemExit('exists ' + str(out))
ev = json.loads((base / 'evaluation/result.json').read_text()); scene = ev['execution']['scene']
fps = 20.0; n = 348; ft = np.arange(n) / fps; rec = json.loads(REC.read_text()); phi_ref = np.zeros(n)
for k, v in rec['phi_deg_per_frame'].items(): phi_ref[int(k)] = v
tr = np.load(ev['execution']['trajectory']); si = [str(x) for x in tr['object_names']].index('wooden_spoon')
st = tr['substep_time_s']; sc = tr['substep_ctrl']
if a.profile:
    kn = sorted((float(x.split(':')[0]), float(x.split(':')[1])) for x in a.profile.split(',')); kf = np.array([k[0] for k in kn]) / fps; kv = np.array([k[1] for k in kn])
    phi = np.zeros(len(st)); m_ = (st >= kf[0]) & (st <= kf[-1]); phi[m_] = PchipInterpolator(kf, kv)(st[m_])
    if a.hold_end: phi[st > kf[-1]] = kv[-1]
else: phi = a.gain * np.interp(st + a.lead, ft, phi_ref)
ps = np.stack([np.interp(st, ft, tr['object_positions'][:n, si, i]) for i in range(3)], 1)
Rs = Slerp(ft, R.from_quat(tr['object_quaternions_wxyz'][:n, si][:, [1, 2, 3, 0]]))(np.clip(st, ft[0], ft[-1]))
with np.load(base / 'training/policy.npz', allow_pickle=False) as zz: d = {k: zz[k] for k in zz.files}
na = [str(x) for x in d['actuator_names']]; pre = 'robot_right_right_'
ids = [na.index(pre + k + '_position') for k in ('pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z')]
act = np.abs(phi) > 1e-9; p0 = sc[:, ids[:3]]; ang0 = sc[:, ids[3:]]; Rw = R.from_euler('XYZ', ang0); p1 = p0.copy(); ang1 = ang0.copy()
if act.any():
    if a.axis == 'wristz':
        C = Rw * R.from_euler('z', phi[:, None], degrees=True) * Rw.inv(); p1 = ps + C.apply(p0 - ps); ang1[:, 2] += np.radians(phi)
    else:
        C = Rs * R.from_euler('x', phi[:, None], degrees=True) * Rs.inv(); p1 = ps + C.apply(p0 - ps); R1 = C * Rw; prev = None
        for k in np.flatnonzero(act):
            anchor = ang0[k] if prev is None or k - prev > 1 else ang1[prev] + (ang0[k] - ang0[prev])
            ang1[k] = continuous_euler_xyz(R.from_matrix(R1[k].as_matrix()[None]), anchor)[0]; prev = k
delta = np.concatenate([p1 - p0, ang1 - ang0], 1); delta[~act] = 0
off = np.array([float(x) for x in a.offset.split(',')]) / 100
smo = lambda u: np.clip(u, 0, 1) ** 3 * (10 - 15 * np.clip(u, 0, 1) + 6 * np.clip(u, 0, 1) ** 2)
og = smo((st - a.off_t0) / a.off_ramp) * (1 - smo((st - (a.off_t1 - a.off_ramp)) / a.off_ramp)); delta[:, :3] += og[:, None] * off; act = act | (og > 0)
sm = lambda u: np.clip(u, 0, 1) ** 3 * (10 - 15 * np.clip(u, 0, 1) + 6 * np.clip(u, 0, 1) ** 2)
gate = sm((st - a.sq_t0) / a.sq_ramp) * (1 - sm((st - (a.sq_t1 - a.sq_ramp)) / a.sq_ramp))
pinch = [na.index(pre + j + '_position') for j in ('thumb_CMC_FE', 'thumb_MCP_FE', 'thumb_IP', 'index_PIP', 'index_DIP')]
wrap = [na.index(pre + f + j + '_position') for f in ('middle_', 'ring_', 'pinky_') for j in ('MCP_FE', 'PIP', 'DIP')]
fids, fdel = [], []
for jl, amt in ((pinch, a.squeeze), (wrap, a.close3)):
    if amt:
        for j in jl: fids.append(j); fdel.append(np.clip(sc[:, j] + amt * gate, d['ctrl_lo'][j], d['ctrl_hi'][j]) - sc[:, j])
allids = ids + fids; D_all = np.concatenate([delta] + ([np.stack(fdel, 1)] if fdel else []), 1)
if a.fb_comp:
    selfg = np.array([d['feedback_weights'][j, j] / d['state_scale'][j] * d['action_scale'][j] for j in allids]); D_all = D_all * (1 - selfg)
use = (np.abs(D_all) > 1e-12).any(1)
Phi = features(st / float(d['duration_s']), d['centers'], float(d['width'])); target = D_all / d['action_scale'][allids]
active = np.nonzero(Phi[use].max(0) > 1e-6)[0] if use.any() else np.array([], int)
A = Phi[:, active]; Dw = np.linalg.solve(A.T @ A + a.ridge * np.eye(len(active)), A.T @ target)
W = d['phase_weights'].copy(); W[np.ix_(active, allids)] += Dw; d['phase_weights'] = W; perr = np.abs(A @ Dw * d['action_scale'][allids] - D_all).max(0)
(out / 'training').mkdir(parents=True, exist_ok=True); ck = out / 'training/policy.npz'; np.savez_compressed(ck, **d)
rep = json.loads((base / 'training/training_report.json').read_text())
rep.update(checkpoint_sha256=sha(ck), roll_refit2=dict(base_actor=str(base / 'training/policy.npz'), base_sha256=sha(base / 'training/policy.npz'), base_rollout=ev['execution']['trajectory'],
    roll_correction=str(REC), roll_correction_sha256=sha(REC), profile=a.profile, gain=a.gain, lead_s=a.lead, axis=a.axis, squeeze_rad=a.squeeze, close3_rad=a.close3, squeeze_window=[a.sq_t0, a.sq_t1, a.sq_ramp], hold_end=a.hold_end, fb_comp=a.fb_comp, offset_cm=a.offset, offset_window=[a.off_t0, a.off_t1, a.off_ramp],
    ridge=a.ridge, phase_centers_changed=int(len(active)), max_wrist_change=dict(pos_m=float(np.abs(delta[:, :3]).max()), rot_rad=float(np.abs(delta[:, 3:]).max())),
    max_rot_joint_speed_rad_s=float(np.abs(np.diff(ang1, axis=0) / np.diff(st)[:, None]).max()), projection_max_error=perr.tolist(), script=str(Path(__file__).resolve()), script_sha256=sha(__file__)))
(out / 'training/training_report.json').write_text(json.dumps(rep, indent=2, default=str))
print('refit', json.dumps({k: rep['roll_refit2'][k] for k in ('max_wrist_change', 'max_rot_joint_speed_rad_s')}), 'proj err max', round(float(perr.max()), 4), flush=True)
if a.no_eval: sys.exit()
from evaluate_policy_component import evaluate
evaluate(33, scene, str(ck), str(out / 'evaluation'), str(OEP))
r = json.loads((out / 'evaluation/result.json').read_text()); m = r['metrics']['official_style_reference_proxy']
sys.path.insert(0, str(P / 'code')); os.environ['RESCORE_EPD'] = str(B / 'episodes/episode_000033'); import importlib, rescore; importlib.reload(rescore)
orig = rescore.score(r['execution']['trajectory'], '/data1/SRL_V2D/challenge_reconstruction/final_recon/episode_000033/object_poses.npz')
# window diagnostics vs corrected reference
man = json.loads((OEP / 'input_manifest.json').read_text()); S = np.asarray(man['setup_from_source']); zc = np.load(Path(man['source']) / 'object_poses.npz'); Pc = S @ zc['world_from_object'][:, 1]
t2 = np.load(r['execution']['trajectory']); Ra = R.from_quat(t2['object_quaternions_wxyz'][:, 1][:, [1, 2, 3, 0]]); er = np.degrees((Ra * R.from_matrix(Pc[:, :3, :3]).inv()).magnitude())
pe = np.linalg.norm(t2['object_positions'][:, 1] - Pc[:, :3, 3], axis=1) * 100; F = t2['hand_object_force_n'][:, 1]
held = int(np.argmax(pe > 6)) if (pe > 6).any() else n
print('DONE', out.name, json.dumps(dict(mp_sr=m['mp_sr'], corrected={k: (round(v['mean_position_error_cm'][0], 3), round(v['mean_orientation_error_degrees'][0], 2), round(v['max_position_error_m'][0] * 100, 1)) for k, v in m['objects'].items()}, original=orig,
    win_rot_deg_f130_170=round(float(er[130:171].mean()), 1), win_pos_cm=round(float(pe[130:171].mean()), 2), lost_at_s=held / fps, Rforce_win_mean=round(float(F[130:171].mean()), 2))), flush=True)
