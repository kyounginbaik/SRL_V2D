#!/usr/bin/env python3
from pathlib import Path
import argparse,json
from contact_acquisition_runtime import execute_acquisition
from evaluate_mp_sr import evaluate_files

B=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--opposing-count',type=int,required=True);a=p.parse_args()
ep='episode_000003';root=B/'refinement_20261017/contact_acquisition'/ep
config=json.loads((root/f'config_{a.opposing_count}opp.json').read_text())
scene=B/'refinement_20261017/cem_v2'/ep/'execution/scene_physics.xml'
reference=B/'refinement_20261017/geometry_combo_v8_clamp'/ep/'tracking_reference_geometry.npz'
out=root/f'feedback_{a.opposing_count}opp';out.mkdir(parents=True,exist_ok=True)
execution=execute_acquisition(scene,reference,out/'execution',config)
inp=B/'episodes'/ep;manifest=json.loads((inp/'input_manifest.json').read_text())
metrics=evaluate_files(manifest['source'],execution['trajectory'],inp/'input_manifest.json',execution['scene'],inp/'metric_geometry.json')
report={'execution':execution,'metrics':metrics,'reference':str(reference),'config':config}
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'report':str(out/'report.json'),'physics_verified':execution['physics_verified'],'events':execution['feedback_events'],'proxy_mp_sr':metrics['reference_proxy_mp_sr']},indent=2))
