"""Episode 30: add extra right-wrist lift (on top of the selected teacher reference) so the sweeper clears the larger scan rack.
usage: extra_lift.py SRC_RUN OUT HEIGHT START END   (blend in over 0.2 s from START, out over 0.4 s ending at END)"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import sys, json
from pathlib import Path
import numpy as np, mujoco
W = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021'); LEG = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017')
sys.path[:0] = [str(LEG / 'code'), '/data1/SRL_V2D/third_party/DexAgent']
from tool_library.common import tracking as TR
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files
src, out = Path(sys.argv[1]), Path(sys.argv[2]); h, t0, t1 = map(float, sys.argv[3:6])
prev = json.loads((src / 'result.json').read_text()); scene = prev['execution']['scene']; inp = Path(os.environ.get('EPDIR', str(W / 'episodes/episode_000030')))
m = mujoco.MjModel.from_xml_path(scene); ref = TR.load_reference(src / 'reference.npz', m); q = ref['qpos'].copy()
b = lambda x: np.clip(x, 0, 1) ** 3 * (10 - 15 * np.clip(x, 0, 1) + 6 * np.clip(x, 0, 1) ** 2)
t = np.arange(len(q)) / ref['fps']; w = b((t - t0) / .2) * b((t1 - t) / .4); a = int(m.joint('robot_right_right_pos_z').qposadr[0]); q[:, a] += h * w
out.mkdir(parents=True, exist_ok=True)
refp = TR.save_reference(out / 'reference.npz', m, q, ref['labels'], ref['fps'], ref['segment'], ref['segments'], source='extra early right-wrist lift to clear the scan-derived rack; other columns unchanged', floating_runtime=True, no_retiming=True)
ex = execute_reference(scene, refp, out / 'execution'); man = json.loads((inp / 'input_manifest.json').read_text())
met = evaluate_files(man['source'], ex['trajectory'], inp / 'input_manifest.json', ex['scene'], inp / 'metric_geometry.json')
res = dict(episode='episode_000030', input_dir=str(inp), reference=str(refp), execution=ex, metrics=met, parent=str(src), extra_lift=dict(height_m=h, start_s=t0, end_s=t1))
(out / 'result.json').write_text(json.dumps(res, indent=1, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)) + '\n')
print(json.dumps(dict(out=str(out), mp_sr=met['reference_proxy_mp_sr'], objects={k: (round(v['mean_position_error_cm'][0], 2), round(v['mean_orientation_error_degrees'][0], 1)) for k, v in met['official_style_reference_proxy']['objects'].items()})), flush=True)
