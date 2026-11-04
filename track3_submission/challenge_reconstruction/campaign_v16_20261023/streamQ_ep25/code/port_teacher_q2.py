"""Stream Q: execute a teacher's recorded robot commands unchanged in the corrected-reference world (objects start from the corrected
scene home) and score against the corrected reference. OUT/reference.npz = teacher reference.npz with ONLY the object target columns
replaced by the corrected tracking reference (robot columns byte-identical), so retention/distill tools accept OUT as --source.
usage: port_teacher_q2.py TEACHER_DIR WORLD_DIR OUT_DIR [SCENE]  (SCENE: e.g. the unchanged v19 scene for world w3b)"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import sys, json
from pathlib import Path
import numpy as np, mujoco
LEG = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017')
sys.path[:0] = [str(LEG / 'code'), '/data1/SRL_V2D/third_party/DexAgent']
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files
from tool_library.common import tracking as TR
teacher, world, out = (Path(x).resolve() for x in sys.argv[1:4]); epdir = world / 'episodes/episode_000025'; scene = Path(sys.argv[4]) if len(sys.argv) > 4 else world / 'scene/scene_physics_corrected.xml'
man = json.loads((epdir / 'input_manifest.json').read_text()); tr = json.loads((teacher / 'result.json').read_text())
src_ref = teacher / 'reference.npz'
out.mkdir(parents=True, exist_ok=False)
z = {k: v for k, v in np.load(src_ref, allow_pickle=False).items()}; m = mujoco.MjModel.from_xml_path(str(scene))
names = list(np.asarray(z['joint_names']).astype(str)); tgt = TR.load_reference(epdir / 'tracking_reference.npz', m)['qpos']
from tool_library.common import episode as E
wanted = E.joint_names(m); Q = np.asarray(z['qpos'], float).copy(); cols = []
for o in ('wooden_piece_1', 'wooden_piece_2'):
    adr = int(m.joint(o + '_free').qposadr[0])
    # free joint occupies 7 qpos entries; reference qpos is in joint-name order of the file -> map via qpos addresses of the file order
    cols.append((o, adr))
# reference qpos layout equals the model qpos layout when joint names are in model order (asserted)
assert names == wanted, 'reference joint order differs from the scene'
for o, adr in cols: Q[:, adr:adr + 7] = tgt[:, adr:adr + 7]
mask = np.ones(Q.shape[1], bool)
for o, adr in cols: mask[adr:adr + 7] = False
assert np.array_equal(Q[:, mask], np.asarray(z['qpos'], float)[:, mask])
z['qpos'] = Q; np.savez(out / 'reference.npz', **z)
ex = execute_reference(str(scene), str(out / 'reference.npz'), out / 'execution')
met = evaluate_files(man['source'], ex['trajectory'], epdir / 'input_manifest.json', ex['scene'], epdir / 'metric_geometry.json')
fmt = lambda mm: {k: (round(v['mean_position_error_cm'][0], 3), round(v['mean_orientation_error_degrees'][0], 2), round(v['max_position_error_m'][0] * 100, 1)) for k, v in mm['official_style_reference_proxy']['objects'].items()}
res = dict(episode='episode_000025', input_dir=str(epdir), reference=str(out / 'reference.npz'), execution=ex, metrics=met, parent=str(teacher), teacher_metrics=tr.get('metrics'),
           port=dict(world=str(world), scene=str(scene), note='teacher robot commands executed unchanged in the corrected world; object target columns = corrected reference'),
           object_targets_unchanged=False, learned_policy=False)
(out / 'result.json').write_text(json.dumps(res, indent=1, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)) + '\n')
print(json.dumps(dict(out=str(out), mp_sr=met['reference_proxy_mp_sr'], objects=fmt(met), verified=ex['physics_verified'])), flush=True)
