"""Episode 30: the idle LEFT hand (HaWoR prior) rests its thumb on top of the rack from ~1 s on and slowly pushes the rack
(0.5 -> 1.6 cm over the episode). Shift the left wrist reference by a constant world offset (blended in over 0.4 s from T_ON);
right hand, fingers and object columns unchanged. Scores in the world given by env EPDIR.
usage: left_offset.py SRC_RUN OUT DX DY DZ [T_ON=0.2]"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import sys, json
from pathlib import Path
import numpy as np, mujoco
LEG = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017')
sys.path[:0] = [str(LEG / 'code'), '/data1/SRL_V2D/third_party/DexAgent']
from tool_library.common import tracking as TR
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files
src, out = Path(sys.argv[1]), Path(sys.argv[2]); off = np.array(list(map(float, sys.argv[3:6]))); t_on = float(sys.argv[6]) if len(sys.argv) > 6 else .2
prev = json.loads((src / 'result.json').read_text()); scene = prev['execution']['scene']; inp = Path(os.environ['EPDIR'])
m = mujoco.MjModel.from_xml_path(scene); ref = TR.load_reference(src / 'reference.npz', m); q = ref['qpos'].copy(); orig = q.copy()
b = lambda x: np.clip(x, 0, 1) ** 3 * (10 - 15 * np.clip(x, 0, 1) + 6 * np.clip(x, 0, 1) ** 2)
t = np.arange(len(q)) / ref['fps']; w = b((t - t_on) / .4)
cols = [int(m.joint(f'robot_left_left_pos_{a}').qposadr[0]) for a in 'xyz']
for c, o in zip(cols, off): q[:, c] += o * w
robot = m.jnt_qposadr[m.actuator_trnid[:, 0]]; q[:, robot] = np.clip(q[:, robot], m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])
assert np.array_equal(np.delete(q, cols, 1), np.delete(orig, cols, 1))
out.mkdir(parents=True, exist_ok=False)
refp = TR.save_reference(out / 'reference.npz', m, q, ref['labels'], ref['fps'], ref['segment'], ref['segments'], source=f'constant left-wrist offset {off.tolist()} m from {t_on} s (idle hand moved off the rack); other columns unchanged', floating_runtime=True, no_retiming=True)
ex = execute_reference(scene, refp, out / 'execution'); man = json.loads((inp / 'input_manifest.json').read_text())
met = evaluate_files(man['source'], ex['trajectory'], inp / 'input_manifest.json', ex['scene'], inp / 'metric_geometry.json')
res = dict(episode='episode_000030', input_dir=str(inp), reference=str(refp), execution=ex, metrics=met, parent=str(src), left_offset=dict(offset_m=off.tolist(), t_on_s=t_on), object_targets_unchanged=True, learned_policy=False)
(out / 'result.json').write_text(json.dumps(res, indent=1, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)) + '\n')
print(json.dumps(dict(out=str(out), mp_sr=met['reference_proxy_mp_sr'], objects={k: (round(v['mean_position_error_cm'][0], 2), round(v['mean_orientation_error_degrees'][0], 1)) for k, v in met['official_style_reference_proxy']['objects'].items()})), flush=True)
