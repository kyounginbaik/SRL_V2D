from pathlib import Path
import argparse,json
from contact_retention_fingers_runtime import execute_retention
from evaluate_mp_sr import evaluate_files

B=Path(__file__).resolve().parents[1];R=B/'refinement_20261017'
p=argparse.ArgumentParser();p.add_argument('--preload',type=float,required=True);a=p.parse_args()
ep='episode_000003';config=json.loads((R/'contact_retention_measured'/ep/'geometry/config.json').read_text())
config['transport_blend_s']=.8;config['capture_finger_preload_m']=a.preload
reference=R/'geometry_combo_v8_clamp'/ep/'tracking_reference_geometry.npz'
scene=R/'cem_v2'/ep/'execution/scene_physics.xml'
out=R/'contact_retention_fingers'/ep/f'preload_{round(a.preload*1000):02d}';out.mkdir(parents=True,exist_ok=True)
(out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
execution=execute_retention(scene,reference,out/'execution',config)
inp=B/'episodes'/ep;manifest=json.loads((inp/'input_manifest.json').read_text())
metrics=evaluate_files(manifest['source'],execution['trajectory'],inp/'input_manifest.json',execution['scene'],inp/'metric_geometry.json')
report={'execution':execution,'metrics':metrics,'reference':str(reference),'config':config}
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'report':str(out/'report.json'),'physics_verified':execution['physics_verified'],'events':execution['feedback_events'],'proxy_mp_sr':metrics['reference_proxy_mp_sr']},indent=2))
