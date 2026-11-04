"""Episode 30: slow down the RIGHT hand's early approach by time-warping its reference columns (wrist + fingers) before T1:
new_q(t) = q(t - DELAY * s(t)), s = 1 for t <= T0, smooth ramp to 0 at T1 (identical after T1). The fast 0.1-0.5 s approach of the
early-steady pickup glances the brush at 0.3-0.4 s and kicks the 100 g rack by ~1 cm; a slower approach should not.
Left hand and object columns unchanged. Scores in the world given by env EPDIR.
usage: timewarp_right.py SRC_RUN OUT DELAY T0 T1"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import sys, json
from pathlib import Path
import numpy as np, mujoco
LEG = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017')
sys.path[:0] = [str(LEG / 'code'), '/data1/SRL_V2D/third_party/DexAgent']
from tool_library.common import hand as H, tracking as TR
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files
src, out = Path(sys.argv[1]), Path(sys.argv[2]); delay, t0, t1 = map(float, sys.argv[3:6])
prev = json.loads((src / 'result.json').read_text()); scene = prev['execution']['scene']; inp = Path(os.environ['EPDIR'])
m = mujoco.MjModel.from_xml_path(scene); ref = TR.load_reference(src / 'reference.npz', m); q = ref['qpos'].copy(); orig = q.copy(); fps = float(ref['fps'])
rig = H.HandRig.build(m, 'robot_right_', 'right'); cols = np.r_[[rig.wrist_qpos[j] for j in ['x', 'y', 'z', 'rx', 'ry', 'rz']], list(rig.finger_qpos.values())]
t = np.arange(len(q)) / fps; u = np.clip((t - t0) / (t1 - t0), 0, 1); s = 1 - u * u * (3 - 2 * u); tw = np.maximum(t - delay * s, 0.)
for c in cols: q[:, c] = np.interp(tw, t, orig[:, c])
assert np.array_equal(q[t >= t1], orig[t >= t1]) and np.array_equal(np.delete(q, cols, 1), np.delete(orig, cols, 1))
out.mkdir(parents=True, exist_ok=False)
refp = TR.save_reference(out / 'reference.npz', m, q, ref['labels'], ref['fps'], ref['segment'], ref['segments'], source=f'right-hand approach time-warp: delay {delay} s until {t0} s, blended out by {t1} s; other columns unchanged', floating_runtime=True, no_retiming=True)
ex = execute_reference(scene, refp, out / 'execution'); man = json.loads((inp / 'input_manifest.json').read_text())
met = evaluate_files(man['source'], ex['trajectory'], inp / 'input_manifest.json', ex['scene'], inp / 'metric_geometry.json')
res = dict(episode='episode_000030', input_dir=str(inp), reference=str(refp), execution=ex, metrics=met, parent=str(src), timewarp_right=dict(delay_s=delay, t0_s=t0, t1_s=t1), object_targets_unchanged=True, learned_policy=False)
(out / 'result.json').write_text(json.dumps(res, indent=1, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)) + '\n')
print(json.dumps(dict(out=str(out), mp_sr=met['reference_proxy_mp_sr'], objects={k: (round(v['mean_position_error_cm'][0], 2), round(v['mean_orientation_error_degrees'][0], 1)) for k, v in met['official_style_reference_proxy']['objects'].items()})), flush=True)
