"""X8: replay a teacher reference (qpos track, no geometry) in its own frozen scene with the X8 cup collision swapped in
(swap_scene_x8.py), score against the X8 episode dir (= world_up targets).  Teacher dir layout output (reference.npz, result.json,
execution/) so kcem_n8.py / actor_grid_c.py can use it as a base / teacher.
usage: replay_x8.py TEACHER_DIR WORLD OUT_DIR"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import sys, json, subprocess, shutil
from pathlib import Path
import numpy as np
R0 = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017'); sys.path[:0] = [str(R0 / 'code'), '/data1/SRL_V2D/third_party/DexAgent']
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files
te, W, out = (Path(p).resolve() for p in sys.argv[1:4]); out.mkdir(parents=True, exist_ok=False)
epd = W / 'episode_target/episode_000010'; man = json.loads((epd / 'input_manifest.json').read_text())
r = json.loads((te / 'result.json').read_text()); old = Path(r['execution']['scene'])
scene = out / 'scene_x8.xml'
subprocess.run([sys.executable, str(Path(__file__).parent / 'swap_scene_x8.py'), str(old), str(W), str(scene)], check=True)
shutil.copy2(te / 'reference.npz', out / 'reference.npz')
ex = execute_reference(str(scene), str(out / 'reference.npz'), out / 'execution')
met = evaluate_files(man['source'], ex['trajectory'], epd / 'input_manifest.json', ex['scene'], epd / 'metric_geometry.json')
res = dict(episode='episode_000010', input_dir=str(epd), parent=str(te), reference=str(out / 'reference.npz'), execution=ex, metrics=met, learned_policy=False,
           note='X8 replay of the teacher reference with the X8 thinner cup collision swapped into its frozen scene',
           spec=dict(episode='episode_000010', scene=str(scene), reference=str(out / 'reference.npz'), source=man['source'], manifest=str(epd / 'input_manifest.json'),
                     geometry=str(epd / 'metric_geometry.json'), original_reference=str(epd / 'tracking_reference.npz')))
(out / 'result.json').write_text(json.dumps(res, indent=1, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)) + '\n')
print('RESULT', met['reference_proxy_mp_sr'], {k: (round(v['mean_position_error_cm'][0], 2), round(v['mean_orientation_error_degrees'][0], 1)) for k, v in met['official_style_reference_proxy']['objects'].items()}, flush=True)
