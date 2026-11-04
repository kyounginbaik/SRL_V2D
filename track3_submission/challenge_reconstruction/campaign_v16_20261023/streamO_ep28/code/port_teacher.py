"""Stream O: execute one teacher's exact recorded reference (robot columns; objects start from the world's scene home) in a
Stream-O world and score it against that world's reference track. Writes result.json in the usual tool schema so that
retention.py / release_v9.py / extra_lift.py / actor_grid_c.py accept the output dir as --source.
usage: port_teacher.py TEACHER_DIR WORLD_DIR OUT_DIR"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import sys, json
from pathlib import Path
import numpy as np
LEG = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017')
sys.path[:0] = [str(LEG / 'code'), '/data1/SRL_V2D/third_party/DexAgent']
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files
teacher, world, out = (Path(x).resolve() for x in sys.argv[1:4]); epdir = world / 'episodes/episode_000028'; scene = world / 'scene/controller_scene.xml'
man = json.loads((epdir / 'input_manifest.json').read_text()); tr = json.loads((teacher / 'result.json').read_text())
ref = Path(tr['reference']) if 'reference' in tr and Path(tr['reference']).exists() else teacher / 'reference.npz'
out.mkdir(parents=True, exist_ok=False)
ex = execute_reference(str(scene), str(ref), out / 'execution')
met = evaluate_files(man['source'], ex['trajectory'], epdir / 'input_manifest.json', ex['scene'], epdir / 'metric_geometry.json')
fmt = lambda m: {k: (round(v['mean_position_error_cm'][0], 2), round(v['mean_orientation_error_degrees'][0], 1), round(v['max_position_error_m'][0] * 100, 1)) for k, v in m['official_style_reference_proxy']['objects'].items()}
res = dict(episode='episode_000028', input_dir=str(epdir), reference=str(ref), execution=ex, metrics=met, parent=str(teacher), teacher_metrics=tr.get('metrics'),
           port=dict(world=str(world), scene=str(scene), note='teacher reference executed unchanged in the Stream-O world; objects start from the world scene home'),
           object_targets_unchanged=True, learned_policy=False)
(out / 'result.json').write_text(json.dumps(res, indent=1, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)) + '\n')
print(json.dumps(dict(out=str(out), mp_sr=met['reference_proxy_mp_sr'], objects=fmt(met), teacher_objects=fmt(tr['metrics']) if tr.get('metrics') else None, verified=ex['physics_verified'])), flush=True)
