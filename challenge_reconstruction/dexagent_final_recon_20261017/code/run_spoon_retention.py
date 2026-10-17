#!/usr/bin/env python3
from pathlib import Path
import argparse,json
from contact_retention_runtime import execute_retention
from evaluate_mp_sr import evaluate_files

B=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--qualified',type=int,choices=[0,1],required=True);a=p.parse_args()
ep='episode_000003'
config=json.loads((B/'refinement_20261017/contact_acquisition'/ep/'config_1opp.json').read_text())
config.update(release_start_s=14.45,release_blend_s=.35,require_opposed_contact_geometry=bool(a.qualified))
scene=B/'refinement_20261017/cem_v2'/ep/'execution/scene_physics.xml'
reference=B/'refinement_20261017/geometry_combo_v8_clamp'/ep/'tracking_reference_geometry.npz'
out=B/'refinement_20261017/contact_retention'/ep/f'qualified_{a.qualified}'
out.mkdir(parents=True,exist_ok=True)
execution=execute_retention(scene,reference,out/'execution',config)
inp=B/'episodes'/ep;manifest=json.loads((inp/'input_manifest.json').read_text())
metrics=evaluate_files(manifest['source'],execution['trajectory'],inp/'input_manifest.json',execution['scene'],inp/'metric_geometry.json')
report={'execution':execution,'metrics':metrics,'reference':str(reference),'config':config}
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'report':str(out/'report.json'),'physics_verified':execution['physics_verified'],'events':execution['feedback_events'],'proxy_mp_sr':metrics['reference_proxy_mp_sr']},indent=2))
