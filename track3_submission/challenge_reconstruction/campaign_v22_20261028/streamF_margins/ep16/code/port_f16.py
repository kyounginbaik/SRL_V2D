"""Stream F ep16: port a streamN teacher into world_T: robot columns of its reference.npz unchanged, object free-joint columns replaced by
world_T's tracking_reference (= target_final_v5 re-target; required by the CEM tools), executed in the unchanged controller scene,
scored against world_T's source (target). Also prints the tlib16 scorer-identical errors of the executed trajectory.
usage: port_f16.py TEACHER_DIR WORLD_DIR OUT_DIR"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import sys, json, shutil
from pathlib import Path
import numpy as np, mujoco
LEG = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017')
sys.path[:0] = [str(LEG / 'code'), '/data1/SRL_V2D/third_party/DexAgent']
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files
from tool_library.common import tracking as TR
teacher, world, out = (Path(x).resolve() for x in sys.argv[1:4]); epdir = world / 'episodes/episode_000016'; scene = world / 'scene/controller_scene.xml'
man = json.loads((epdir / 'input_manifest.json').read_text()); tr = json.loads((teacher / 'result.json').read_text())
ref = Path(tr['reference']) if 'reference' in tr and Path(tr['reference']).exists() else teacher / 'reference.npz'
out.mkdir(parents=True, exist_ok=False)
m = mujoco.MjModel.from_xml_path(str(scene)); s = TR.load_reference(ref, m); o = TR.load_reference(epdir / 'tracking_reference.npz', m); Q = s['qpos'].copy()
for n in ('plastic_dish_rack', 'wooden_spoon'):
    c = int(m.joint(n + '_free').qposadr[0]); Q[:, c:c + 7] = o['qpos'][:, c:c + 7]
TR.save_reference(out / 'reference.npz', m, Q, s['labels'], s['fps'], s['segment'], s['segments'], source=f'{ref} robot columns + object columns of {epdir} tracking_reference (target_final_v5)', floating_runtime=True, no_retiming=True)
ref = out / 'reference.npz'
ex = execute_reference(str(scene), str(ref), out / 'execution')
met = evaluate_files(man['source'], ex['trajectory'], epdir / 'input_manifest.json', ex['scene'], epdir / 'metric_geometry.json')
fmt = lambda m: {k: (round(v['mean_position_error_cm'][0], 2), round(v['mean_orientation_error_degrees'][0], 1), round(v['max_position_error_m'][0] * 100, 1)) for k, v in m['official_style_reference_proxy']['objects'].items()}
res = dict(episode='episode_000016', input_dir=str(epdir), reference=str(ref), execution=ex, metrics=met, parent=str(teacher), teacher_metrics=tr.get('metrics'),
           port=dict(world=str(world), scene=str(scene), note='streamF: teacher robot columns executed unchanged in the unchanged scene; object targets = target_final_v5 re-target'),
           object_targets_unchanged=False, learned_policy=False)
(out / 'result.json').write_text(json.dumps(res, indent=1, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)) + '\n')
print(json.dumps(dict(out=str(out), mp_sr=met['reference_proxy_mp_sr'], objects=fmt(met), verified=ex['physics_verified'])), flush=True)
