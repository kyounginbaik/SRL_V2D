"""Stream L: replay a teacher reference (qpos track) in fresh physics with OUR thin-wall recon cups swapped into a frozen
recon-world scene (determinism control of the unchanged scene first); score against the stream-L episode dir.
usage: retarget_recon.py SCENE_TEACHER_DIR REF_TEACHER_DIR OUT_DIR
  SCENE_TEACHER_DIR: dir with execution/scene_physics.xml (recon-cup frozen scene, e.g. calibrated_nest_release4)
  REF_TEACHER_DIR  : dir with reference.npz (+ result.json / execution/trajectory.npz for the control)"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import sys, json, subprocess, shutil
from pathlib import Path
import numpy as np
R0 = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017'); EP10 = R0 / 'refinement_20261021_ep10'
S = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamL_ep10'); WORLD = S / 'world_v1'; epdir = WORLD / 'episode_000010'
sys.path[:0] = [str(R0 / 'code'), '/data1/SRL_V2D/third_party/DexAgent']
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files
scene_teacher, ref_teacher, out = (Path(p).resolve() for p in sys.argv[1:4])
man = json.loads((epdir / 'input_manifest.json').read_text()); source = man['source']
tscene = scene_teacher / 'execution/scene_physics.xml'; ref = ref_teacher / 'reference.npz'; out.mkdir(parents=True, exist_ok=True)
scene = out / 'scene_reconcups.xml'; tmp = out / 'scene_tmp.xml'
if not scene.exists():
    subprocess.run([sys.executable, str(EP10 / 'code/swap_object_scene.py'), 'blue_cup', str(tscene), str(epdir / 'stage2/assets/blue_cup'), str(WORLD / 'source_assets/blue_cup.urdf'), str(tmp)], check=True)
    subprocess.run([sys.executable, str(EP10 / 'code/swap_object_scene.py'), 'beige_cup', str(tmp), str(epdir / 'stage2/assets/beige_cup'), str(WORLD / 'source_assets/beige_cup.urdf'), str(scene)], check=True)
txt = scene.read_text(); assert 'gt_best_per_object' not in txt and 'refinement_20261021_ep10' not in txt, 'scan lineage leaked into the scene'
det = None
if (ref_teacher / 'execution/trajectory.npz').exists() and ref_teacher == scene_teacher:
    ctrl = out / 'control_old_scene'
    if not (ctrl / 'execution/trajectory.npz').exists(): execute_reference(str(tscene), str(ref), ctrl / 'execution')
    with np.load(ctrl / 'execution/trajectory.npz') as a, np.load(ref_teacher / 'execution/trajectory.npz') as b: det = float(np.abs(a['qpos'] - b['qpos']).max())
ex = execute_reference(str(scene), str(ref), out / 'execution')
met = evaluate_files(source, ex['trajectory'], epdir / 'input_manifest.json', ex['scene'], epdir / 'metric_geometry.json')
res = dict(episode='episode_000010', input_dir=str(epdir), reference=str(ref), execution=ex, metrics=met, teacher=str(ref_teacher), scene_teacher=str(scene_teacher),
           determinism_control_max_qpos_diff=det, source_targets_unchanged=True, learned_policy=False,
           spec=dict(episode='episode_000010', scene=str(scene), reference=str(ref), source=source, manifest=str(epdir / 'input_manifest.json'), geometry=str(epdir / 'metric_geometry.json'),
                     original_reference=str(epdir / 'tracking_reference.npz')))
(out / 'result.json').write_text(json.dumps(res, indent=1, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)) + '\n')
shutil.copy2(ref, out / 'reference.npz')
o = lambda m: {k: (round(v['mean_position_error_cm'][0], 2), round(v['mean_orientation_error_degrees'][0], 1), round(v['max_position_error_cm'][0] if 'max_position_error_cm' in v else 100 * v['max_position_error_m'][0], 2)) for k, v in m['official_style_reference_proxy']['objects'].items()}
print(json.dumps(dict(determinism=det, mp_sr=met['reference_proxy_mp_sr'], physics_verified=ex.get('physics_verified'), objects=o(met))), flush=True)
