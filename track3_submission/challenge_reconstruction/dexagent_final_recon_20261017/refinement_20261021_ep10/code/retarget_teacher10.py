"""Episode 10: replay the shipped teacher reference in fresh physics with both scan cups swapped into its frozen scene
(determinism control in the unchanged scene first); score against the new episode dir geometry (object targets unchanged).
usage: retarget_teacher10.py TEACHER_DIR OUT_DIR"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import sys, json, subprocess
from pathlib import Path
import numpy as np
R0 = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017'); EP10 = R0 / 'refinement_20261021_ep10'
sys.path[:0] = [str(R0 / 'code'), '/data1/SRL_V2D/third_party/DexAgent']
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files
teacher, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(); epdir = EP10 / 'episode_dir/episode_000010'
man = json.loads((epdir / 'input_manifest.json').read_text()); source = man['source']
tscene = teacher / 'execution/scene_physics.xml'; ref = teacher / 'reference.npz'; out.mkdir(parents=True, exist_ok=True)
scene = out / 'scene_scancups.xml'; tmp = out / 'scene_tmp.xml'
subprocess.run([sys.executable, str(EP10 / 'code/swap_object_scene.py'), 'blue_cup', str(tscene), str(epdir / 'stage2/assets/blue_cup'), str(EP10 / 'blue_cup.urdf'), str(tmp)], check=True)
subprocess.run([sys.executable, str(EP10 / 'code/swap_object_scene.py'), 'beige_cup', str(tmp), str(epdir / 'stage2/assets/beige_cup'), str(EP10 / 'beige_cup.urdf'), str(scene)], check=True)
ctrl = out / 'control_old_scene'
if not (ctrl / 'execution/trajectory.npz').exists(): execute_reference(str(tscene), str(ref), ctrl / 'execution')
with np.load(ctrl / 'execution/trajectory.npz') as a, np.load(teacher / 'execution/trajectory.npz') as b: det = float(np.abs(a['qpos'] - b['qpos']).max())
ex = execute_reference(str(scene), str(ref), out / 'execution')
met = evaluate_files(source, ex['trajectory'], epdir / 'input_manifest.json', ex['scene'], epdir / 'metric_geometry.json')
tr = json.loads((teacher / 'result.json').read_text())
res = dict(episode='episode_000010', input_dir=str(epdir), reference=str(ref), execution=ex, metrics=met, teacher=str(teacher), determinism_control_max_qpos_diff=det,
           spec=dict(episode='episode_000010', scene=str(scene), reference=str(ref), source=source, manifest=str(epdir / 'input_manifest.json'), geometry=str(epdir / 'metric_geometry.json'),
                     original_reference=str(epdir / 'tracking_reference.npz')))
(out / 'result.json').write_text(json.dumps(res, indent=1, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)) + '\n')
import shutil; shutil.copy2(ref, out / 'reference.npz')
o = lambda m: {k: (round(v['mean_position_error_cm'][0], 2), round(v['mean_orientation_error_degrees'][0], 1)) for k, v in m['official_style_reference_proxy']['objects'].items()}
print(json.dumps(dict(determinism=det, mp_sr=met['reference_proxy_mp_sr'], objects=o(met), teacher_objects=o(tr['metrics']))), flush=True)
