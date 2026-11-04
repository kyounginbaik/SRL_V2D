"""Publish an explicitly incomplete reconstruction for the native final checks."""
from pathlib import Path
import json
import shutil

workspace=Path.cwd()
def load(name):return json.loads(Path(name).read_text())
Path('masks').mkdir(exist_ok=True)
shutil.copy2('source_masks/rack/mask.png','masks/rack.png')
shutil.copy2('scene_rack.xml','scene.xml')
if Path('camera.json').exists():shutil.copy2('camera.json','camera_unanchored.json')
shutil.copy2('camera_table_edge.json','camera.json')
shutil.copy2('rack_compare.png','frame0_compare.png')
fit=load('rack_placement.json')
visual=load('rack_visual_checks.json')
rest=load('rack_rest_checks.json')
verifiers={**visual,**rest}
for value in verifiers.values():value.pop('settled_qpos',None)
sem=load('../stage1/scene_semantics.json')
rack=next(o for o in sem['objects'] if o['id']=='rack')
coverage={rack['properties'][0]:'The parameterized rack has two rows of six true openings.',
    rack['properties'][1]:'NOT VERIFIED: no source-grounded tube model or withdrawal execution exists.',
    rack['properties'][2]:'Native support_check and settle_stability_check pass for the stationary rack.'}
manifest={'scene_semantic':'../stage1/scene_semantics.json','scene_xml':'scene.xml','camera_json':'camera.json',
    'frame0':'source_masks/rack/source.png','frame0_compare_png':'frame0_compare.png','simulator':'mujoco',
    'status':'incomplete candidate; not accepted',
    'objects':[{'id':'rack','type':'rigid','skill':'construct_conical_tube_rack','skill_origin':'library',
        'params':load('assets/rack/rack_params.json'),'asset_xml':'assets/rack/rack.xml',
        'root_body':'rack_rack','bodies':['rack_rack'],'pose':fit['pose'],'mask':'masks/rack.png',
        'iou':visual['silhouette_iou_check']['iou']['rack_rack'],
        'verification_plan':{k:'Actual rack placement or stationary support diagnostic; no tube interaction certification.' for k in verifiers},
        'verifiers':verifiers,'properties_covered':coverage}],
    'notes':['The test_tube Stage 1 object is missing: both CPU and SAM3 attempts failed to isolate its observed stem.',
        'An empty initial mask cannot justify a hidden declaration because SAM3 also returns an empty mask in a view where the tube is visibly present.',
        'The nominal separate hand/arm diagnostics do not certify this source reconstruction.',
        'No goal, verifier, robot station, collision rule or motor limit has been weakened.']}
Path('scene_recon.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps({'partial_objects':['rack'],'missing_objects':['test_tube'],'accepted':False}))
