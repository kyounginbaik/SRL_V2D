"""E9 (copy of U8 mkbase.py, label v6): write a staging BASE (actor_report.json + input_manifest.json) for a phase-residual actor, derived from
the CURRENT component's staging base (eps 15/16: U8 staging/ep15_c, ep16_b). usage: mkbaseE9.py SRC_BASE NEW_BASE ACTOR_DIR 'selection text'
object errors are taken from the actor's official evaluation re-scored by mresE9 (vs target_final_v6) and passed as json."""
import json, shutil, sys, hashlib
from pathlib import Path
src, base, actor, sel, obj = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]).resolve(), sys.argv[4], json.loads(sys.argv[5])
if base.exists(): raise SystemExit(f'exists: {base}')
rep = json.loads((src / 'actor_report.json').read_text()); ck = actor / 'training/policy.npz'
rep.update(checkpoint=str(ck), evaluation=str(actor / 'evaluation'), training_report=str(actor / 'training/training_report.json'),
           checkpoint_sha256=hashlib.sha256(ck.read_bytes()).hexdigest(), selection=sel, object_errors_vs_target_final_v6=obj, source_actor_report=str(src / 'actor_report.json'))
rep.pop('object_errors_vs_target_final_v5', None); rep['object_errors'] = obj
r = json.loads((actor / 'evaluation/result.json').read_text()); assert r['reference_proxy_success'] and r['execution']['physics_verified'], 'actor eval not successful'
base.mkdir(parents=True); (base / 'actor_report.json').write_text(json.dumps(rep, indent=2) + '\n'); shutil.copyfile(src / 'input_manifest.json', base / 'input_manifest.json')
print(json.dumps(rep, indent=1))
