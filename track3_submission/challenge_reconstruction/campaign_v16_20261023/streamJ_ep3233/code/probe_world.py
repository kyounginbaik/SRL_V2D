"""Stream J: replay a frozen teacher reference (reference.npz) in a copy of its scene where the blue_cup geometry is swapped for the
assets in --mesh-dir (<cup>_visual.obj + <cup>_convex/part_*.obj) with inertia from --urdf, and score it against --episode-dir.
Writes OUT/result.json in the retention-tool format (execution.scene / execution.trajectory, reference.npz copied) so OUT can
be the --source of later re-plans.  usage: probe_world.py --teacher DIR --out DIR --episode-dir DIR --mesh-dir DIR --urdf FILE"""
import os, sys, json, shutil, subprocess, argparse
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
from pathlib import Path
import numpy as np
R0 = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017'); sys.path[:0] = [str(R0 / 'code'), '/data1/SRL_V2D/third_party/DexAgent']
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files
ap = argparse.ArgumentParser(); [ap.add_argument('--' + k, type=Path, required=True) for k in ('teacher', 'out', 'episode-dir', 'mesh-dir', 'urdf')]
ap.add_argument('--object', default='blue_cup'); ap.add_argument('--no-swap', action='store_true'); a = ap.parse_args()
teacher, out, epdir = a.teacher.resolve(), a.out.resolve(), a.episode_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
man = json.loads((epdir / 'input_manifest.json').read_text()); source = man['source']
prev = json.loads((teacher / 'result.json').read_text()); tscene = Path(prev['execution']['scene']); ref = teacher / 'reference.npz'
scene = out / 'scene_recon.xml'
if a.no_swap: shutil.copyfile(tscene, scene)
else: subprocess.run([sys.executable, str(R0 / 'refinement_20261021_ep32/code/swap_object_scene.py'), a.object, str(tscene), str(a.mesh_dir), str(a.urdf), str(scene)], check=True)
if not (out / 'reference.npz').exists(): shutil.copyfile(ref, out / 'reference.npz')
ex = execute_reference(str(scene), str(out / 'reference.npz'), out / 'execution')
met = evaluate_files(source, ex['trajectory'], epdir / 'input_manifest.json', ex['scene'], epdir / 'metric_geometry.json')
res = dict(episode=man['episode'], input_dir=str(epdir), reference=str(out / 'reference.npz'), execution=ex, metrics=met, teacher=str(teacher), swapped_object=None if a.no_swap else a.object,
           mesh_dir=str(a.mesh_dir), urdf=str(a.urdf), best=prev.get('best', {}),
           spec=dict(episode=man['episode'], scene=str(scene), reference=str(out / 'reference.npz'), source=source, manifest=str(epdir / 'input_manifest.json'), geometry=str(epdir / 'metric_geometry.json'), original_reference=str(epdir / 'tracking_reference.npz')))
(out / 'result.json').write_text(json.dumps(res, indent=1, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)) + '\n')
o = lambda mm: {k: (round(v['mean_position_error_cm'][0], 2), round(v['mean_orientation_error_degrees'][0], 1), round(v['max_position_error_m'][0] * 100, 1), round(v['max_orientation_error_degrees'][0], 1)) for k, v in mm['official_style_reference_proxy']['objects'].items()}
print(json.dumps(dict(mp_sr=met['reference_proxy_mp_sr'], objects=o(met), teacher_objects=o(prev['metrics']), physics_verified=ex.get('physics_verified'))), flush=True)
