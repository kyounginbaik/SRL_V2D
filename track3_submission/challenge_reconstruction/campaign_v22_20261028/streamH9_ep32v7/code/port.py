"""H9 copy (target_final_v7_g9rot). B9: port an existing ep32 actor (training/policy.npz) into a world built by build_world_b.py (v6): copy the checkpoint, run the
official component evaluator in the new scene/episode dir, score scorer-faithfully vs target_final_v7_g9rot (mlib, == score_any).
usage: port.py SRC_ACTOR_DIR WORLD_DIR OUT_ACTOR_DIR MESHES_DIR"""
import sys, json, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, mres
src, world, out, meshes = map(Path, sys.argv[1:5])
if out.exists(): raise SystemExit('exists')
(out / 'training').mkdir(parents=True); import hashlib; sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
SC = world / 'scene/scene_physics.xml'
with np.load(src / 'training/policy.npz', allow_pickle=False) as z: d = {k: z[k] for k in z.files}
old = str(d['scene_sha256']); d['scene_sha256'] = np.array(sha(SC)); np.savez_compressed(out / 'training/policy.npz', **d)   # as streamP port_actor.py: only the scene binding changes
tr = json.loads((src / 'training/training_report.json').read_text()); tr['b9_port'] = dict(source_actor=str(src), source_checkpoint_sha256=sha(src / 'training/policy.npz'), world=str(world), old_scene_sha256=old, new_scene_sha256=sha(SC),
    note='same weights; only scene_sha256 rebound to the v6 world scene (objects initial pose changed)'); tr['checkpoint_sha256'] = sha(out / 'training/policy.npz')
tr['b9_port']['original_training_fields'] = {k: tr.get(k) for k in ('scene', 'scene_sha256')}; tr.update(scene=str(SC), scene_sha256=sha(SC))   # as streamP port_actor.py (the teacher-scene check of the stager)
(out / 'training/training_report.json').write_text(json.dumps(tr, indent=2))
from evaluate_policy_component import evaluate as ev_final
ev_final(32, str(world / 'scene/scene_physics.xml'), str(out / 'training/policy.npz'), str(out / 'evaluation'), str(world / 'episode_corrected/episode_000032'))
env = mres.Env(dict(ep=32, actor=str(out), meshes=str(meshes)))
q = np.load(out / 'evaluation/execution/trajectory.npz')['qpos']; c, res, e, _ = env.score(q)
r = json.loads((out / 'evaluation/result.json').read_text())
print('PORT', json.dumps(dict(v6_scorer=res, proxy_mp_sr=r['metrics'].get('reference_proxy_mp_sr'), physics_verified=r['execution'].get('physics_verified'))), flush=True)
mres.windows(env, e)
