"""Stream B: make a held spoon follow the target by re-planning the RIGHT wrist commands only after the grasp.
offset = inv(T_spoon_achieved(t_lock)) * T_wrist_cmd(t_lock) measured on the BASE rollout (commanded wrist vs achieved spoon, so actuator
lag at t_lock is folded in); follow command F(t) = T_spoon_target(t) * offset (* optional extra local rotation/translation); command
= slerp(base(t), F(t), w(t)) with w ramping 0->gain over [t_on, t_on+ramp] and back to 0 over [t_off-ramp, t_off]. Optional --lag s:
evaluate the target at t+lag (lead). Fingers / left hand unchanged. Executes in the base's scene and scores vs EPD (proxy).
usage: follow_plan.py BASE_DIR OUT_DIR --epd EPD --t-lock 2.9 --t-on 2.9 --t-off 10.8 [--ramp 0.4] [--gain 1] [--lead 0] [--sigma 0]"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import sys, json, argparse
from pathlib import Path
import numpy as np, mujoco
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import gaussian_filter1d
LEG = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017'); sys.path[:0] = [str(LEG / 'code'), '/data1/SRL_V2D/third_party/DexAgent']
from tool_library.common import tracking as TR
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files
from wrist_reference import continuous_euler_xyz
ap = argparse.ArgumentParser(); ap.add_argument('base'); ap.add_argument('out'); ap.add_argument('--epd', required=True)
ap.add_argument('--t-lock', type=float, required=True); ap.add_argument('--t-on', type=float, required=True); ap.add_argument('--t-off', type=float, required=True)
ap.add_argument('--ramp', type=float, default=0.4); ap.add_argument('--gain', type=float, default=1.0); ap.add_argument('--lead', type=float, default=0.0); ap.add_argument('--sigma', type=float, default=0.0)
a = ap.parse_args(); base = Path(a.base); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); epd = Path(a.epd)
r = json.loads((base / 'result.json').read_text()); scene = r['execution']['scene']; m = mujoco.MjModel.from_xml_path(scene)
ref = TR.load_reference(base / 'reference.npz', m); Q = ref['qpos'].copy(); n = len(Q); fps = ref['fps']; t = np.arange(n) / fps
tr = np.load(r['execution']['trajectory']); names = [str(x) for x in tr['object_names']]; si = names.index('wooden_spoon')
qt = TR.load_reference(epd / 'tracking_reference.npz', m)['qpos']; sc = int(m.joint('wooden_spoon_free').qposadr[0])
adr = [int(m.joint(f'robot_right_right_{j}').qposadr[0]) for j in ('pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z')]
k0 = round(a.t_lock * fps); ps0 = tr['object_positions'][k0, si]; Rs0 = R.from_quat(tr['object_quaternions_wxyz'][k0, si][[1, 2, 3, 0]])
Rw0 = R.from_euler('XYZ', Q[k0, adr[3:]]); off_p = Rs0.inv().apply(Q[k0, adr[:3]] - ps0); off_R = Rs0.inv() * Rw0
sm = lambda u: np.clip(u, 0, 1) ** 3 * (10 - 15 * np.clip(u, 0, 1) + 6 * np.clip(u, 0, 1) ** 2)
w = a.gain * sm((t - a.t_on) / a.ramp) * (1 - sm((t - (a.t_off - a.ramp)) / a.ramp))
kt = np.clip(np.round((t + a.lead) * fps).astype(int), 0, n - 1)
P = qt[kt, sc:sc + 3]; Rq = qt[kt, sc + 3:sc + 7]
if a.sigma > 0:
    P = gaussian_filter1d(P, a.sigma * fps, axis=0, mode='nearest')
    rv = R.from_quat(Rq, scalar_first=True); ref0 = rv[0]; lv = np.unwrap((ref0.inv() * rv).as_rotvec(), axis=0)  # small-angle smoothing only
RT = R.from_quat(Rq, scalar_first=True)
for k in range(1, n):
    if w[k] <= 0: continue
    Fp = P[k] + RT[k].apply(off_p); FR = RT[k] * off_R
    Bp = Q[k, adr[:3]]; BR = R.from_euler('XYZ', Q[k, adr[3:]])
    Q[k, adr[:3]] = (1 - w[k]) * Bp + w[k] * Fp
    Rn = BR * R.from_rotvec((BR.inv() * FR).as_rotvec() * w[k])
    Q[k, adr[3:]] = continuous_euler_xyz(R.from_matrix(Rn.as_matrix()[None]), Q[k - 1, adr[3:]])[0]
refp = TR.save_reference(out / 'reference.npz', m, Q, ref['labels'], fps, ref['segment'], ref['segments'], source=f'follow_plan of {base}: {vars(a)}', arms_coupled=False, floating_runtime=True)
ex = execute_reference(scene, refp, out / 'execution'); mf = epd / 'input_manifest.json'; mm = json.loads(mf.read_text())
mt = evaluate_files(mm['source'], ex['trajectory'], mf, ex['scene'], epd / 'metric_geometry.json'); o = mt['official_style_reference_proxy']
res = dict(episode=mm['episode'], input_dir=str(epd.resolve()), reference=str(refp), execution=ex, metrics=mt, args=vars(a), script=__file__)
(out / 'result.json').write_text(json.dumps(res, indent=1, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)) + '\n')
print('RESULT', o['mp_sr'], {k: (round(v['mean_position_error_cm'][0], 2), round(v['mean_orientation_error_degrees'][0], 1)) for k, v in o['objects'].items()}, flush=True)
