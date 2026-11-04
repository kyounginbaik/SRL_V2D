"""Re-run one rack-episode teacher (its exact recorded reference) in fresh physics with the scan-derived rack swapped into the
teacher's frozen scene, after a determinism control in the unchanged teacher scene. Scores with the new episode geometry.
usage: retarget_teacher.py EPISODE TEACHER_DIR OUT_DIR"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import sys, json, subprocess
from pathlib import Path
import numpy as np
W = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/ep30_v9'); LEG = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017')
sys.path[:0] = [str(LEG / 'code'), '/data1/SRL_V2D/third_party/DexAgent']
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files
ep, teacher, out = int(sys.argv[1]), Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve(); name = f'episode_{ep:06d}'
epdir = W / 'episodes' / name; man = json.loads((epdir / 'input_manifest.json').read_text()); source = man['source']
tr = json.loads((teacher / 'result.json').read_text()); tscene = teacher / 'execution/scene_physics.xml'; ref = teacher / 'reference.npz'
out.mkdir(parents=True, exist_ok=True); scene = out / 'scene_rackscan.xml'
subprocess.run([sys.executable, str(W / 'code/swap_rack_scene.py'), str(tscene), str(epdir / 'stage2/assets/plastic_dish_rack'),
                str(W / 'collision_recon' / name / 'assets/plastic_dish_rack.urdf'), str(scene)], check=True)
ctrl_dir = out / 'control_old_scene'
if not (ctrl_dir / 'execution/trajectory.npz').exists():
    execute_reference(str(tscene), str(ref), ctrl_dir / 'execution')
with np.load(ctrl_dir / 'execution/trajectory.npz') as a, np.load(teacher / 'execution/trajectory.npz') as b:
    determinism = float(np.abs(a['qpos'] - b['qpos']).max())
execution = execute_reference(str(scene), str(ref), out / 'execution')
metrics = evaluate_files(source, execution['trajectory'], epdir / 'input_manifest.json', execution['scene'], epdir / 'metric_geometry.json')
objs = {k: (round(v['mean_position_error_cm'][0], 2), round(v['mean_orientation_error_degrees'][0], 1)) for k, v in metrics['official_style_reference_proxy']['objects'].items()}
result = dict(episode=name, input_dir=str(epdir), reference=str(ref), execution=execution, metrics=metrics, teacher=str(teacher), teacher_metrics=tr.get('metrics'),
              determinism_control_max_qpos_diff=determinism, rack_swap=dict(scene=str(scene), source_scene=str(tscene), assets=str(epdir / 'stage2/assets/plastic_dish_rack')),
              spec=dict(episode=name, scene=str(scene), reference=str(ref), source=source, manifest=str(epdir / 'input_manifest.json'), geometry=str(epdir / 'metric_geometry.json')))
(out / 'result.json').write_text(json.dumps(result, indent=1, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)) + '\n')
print(json.dumps(dict(episode=ep, determinism=determinism, mp_sr=metrics['official_style_reference_proxy']['mp_sr'], reference_proxy_mp_sr=metrics.get('reference_proxy_mp_sr'), objects=objs,
                      teacher_objects={k: (round(v['mean_position_error_cm'][0], 2), round(v['mean_orientation_error_degrees'][0], 1)) for k, v in tr['metrics']['official_style_reference_proxy']['objects'].items()})), flush=True)
