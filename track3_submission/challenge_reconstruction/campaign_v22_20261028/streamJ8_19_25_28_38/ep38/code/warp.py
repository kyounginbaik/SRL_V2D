"""J8 ep38: time-warp the ROBOT commands of a teacher in a window (advance the pickup): Q_new(t) = Q_old(t + s*b(t)), b = smooth bump
(0 at a and b, 1 on [a+r, b-r]); object columns untouched (= target); executes with the unchanged floating runtime in the teacher scene and
evaluates against the J8 world0 episode dir.  usage: warp.py TEACHER OUT --a 1.5 --b 3.5 --shift 0.15 [--ramp 0.5]"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import sys, json, argparse
from pathlib import Path
import numpy as np, mujoco
LEG = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017'); sys.path[:0] = [str(LEG / 'code'), '/data1/SRL_V2D/third_party/DexAgent']
from tool_library.common import tracking as TR
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files
EPD = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamJ8_19_25_28_38/ep38/world0/episode_target/episode_000038')
sm = lambda u: np.clip(u, 0, 1) ** 3 * (10 - 15 * np.clip(u, 0, 1) + 6 * np.clip(u, 0, 1) ** 2)
p = argparse.ArgumentParser(); p.add_argument('teacher'); p.add_argument('out'); p.add_argument('--a', type=float, default=1.5); p.add_argument('--b', type=float, default=3.5)
p.add_argument('--shift', type=float, default=.15); p.add_argument('--ramp', type=float, default=.5); a = p.parse_args()
te, out = Path(a.teacher).resolve(), Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=False)
tr = json.loads((te / 'result.json').read_text()); scene = tr['execution']['scene']; m = mujoco.MjModel.from_xml_path(scene)
ref = TR.load_reference(te / 'reference.npz', m); Q = ref['qpos'].copy(); n = len(Q); fps = ref['fps']; t = np.arange(n) / fps
bump = sm((t - a.a) / a.ramp) * (1 - sm((t - (a.b - a.ramp)) / a.ramp)); tw = np.clip(t + a.shift * bump, 0, t[-1])
assert np.all(np.diff(tw) > 0), 'warp not monotonic'
obj = set()
for j in range(m.njnt):
    if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE: obj.update(range(m.jnt_qposadr[j], m.jnt_qposadr[j] + 7))
cols = [c for c in range(Q.shape[1]) if c not in obj]
Qn = Q.copy(); Qn[:, cols] = np.stack([np.interp(tw, t, Q[:, c]) for c in cols], 1); Qn[0] = Q[0]
rp = TR.save_reference(out / 'reference.npz', m, Qn, ref['labels'], fps, ref['segment'], ref['segments'], source=f'J8 ep38 robot-command time warp of {te} ({vars(a)})', arms_coupled=False, floating_runtime=True)
ex = execute_reference(scene, rp, out / 'execution'); man = json.loads((EPD / 'input_manifest.json').read_text())
met = evaluate_files(man['source'], ex['trajectory'], EPD / 'input_manifest.json', ex['scene'], EPD / 'metric_geometry.json')
res = dict(episode='episode_000038', input_dir=str(EPD), reference=str(rp), execution=ex, metrics=met, parent=str(te), warp=vars(a), learned_policy=False)
(out / 'result.json').write_text(json.dumps(res, indent=1, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)) + '\n')
o = met['official_style_reference_proxy']['objects']['mini_dust_pan']
print(json.dumps(dict(out=out.name, mp_sr=met['reference_proxy_mp_sr'], cm=round(o['mean_position_error_cm'][0], 2), deg=round(o['mean_orientation_error_degrees'][0], 1), max_cm=round(o['max_position_error_m'][0] * 100, 1))), flush=True)
