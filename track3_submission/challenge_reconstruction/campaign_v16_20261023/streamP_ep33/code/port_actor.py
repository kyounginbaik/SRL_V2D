"""Stream P ep33 phase 6: port an existing ep33 actor (weights, margins, release primitives unchanged) into the compose-target
world (staging_C/scene/scene_physics.xml: same robot/meshes/collision, only the two objects' initial poses differ). The checkpoint
copy changes only scene_sha256 (the same field the frozen portable relocation rewrites); the training report copy records the
port (original scene/sha, original checkpoint sha) and names the new scene, then a fresh physical evaluation with the unchanged
evaluate_policy_component against staging_C/episode_compose. usage: port_actor.py ACTOR_DIR OUT_DIR"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import sys, json, hashlib, numpy as np
from pathlib import Path
LEG = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017'); sys.path[:0] = [str(LEG / 'code'), '/data1/SRL_V2D/third_party/DexAgent']
P = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamP_ep33'); SC = P / os.environ.get('CW_OUT', 'staging_C') / 'scene/scene_physics.xml'; OEP = P / os.environ.get('CW_OUT', 'staging_C') / 'episode_compose/episode_000033'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
src, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
if (out / 'evaluation/result.json').exists(): raise SystemExit('done')
with np.load(src / 'training/policy.npz', allow_pickle=False) as z: d = {k: z[k] for k in z.files}
old_scene_sha = str(d['scene_sha256']); d['scene_sha256'] = np.array(sha(SC))
(out / 'training').mkdir(parents=True, exist_ok=True); ck = out / 'training/policy.npz'; np.savez_compressed(ck, **d)
rep = json.loads((src / 'training/training_report.json').read_text()); old = {k: rep.get(k) for k in ('scene', 'scene_sha256', 'checkpoint_sha256')}
rep.update(scene=str(SC), scene_sha256=sha(SC), checkpoint_sha256=sha(ck), port_to_compose_world=dict(source_actor=str(src), source_checkpoint_sha256=sha(src / 'training/policy.npz'), original_scene_sha256=old_scene_sha, original_training_fields=old,
    changed_fields=['scene_sha256 (checkpoint)', 'scene/scene_sha256/checkpoint_sha256 (report)'], weights_unchanged=True, scene_change=json.loads((SC.parent / 'scene_change.json').read_text()), script=str(Path(__file__).resolve()), script_sha256=sha(__file__)))
(out / 'training/training_report.json').write_text(json.dumps(rep, indent=2, default=str))
from evaluate_policy_component import evaluate
r = evaluate(33, str(SC), str(ck), str(out / 'evaluation'), str(OEP)); o = r['metrics']['official_style_reference_proxy']['objects']
print('DONE', out.name, json.dumps({k: (round(v['mean_position_error_cm'][0], 3), round(v['mean_orientation_error_degrees'][0], 2), round(v['max_position_error_m'][0] * 100, 1)) for k, v in o.items()}), flush=True)
