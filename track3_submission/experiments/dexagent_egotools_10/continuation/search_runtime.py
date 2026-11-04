"""Bounded native physical candidate search on the explicitly nominal scene."""
from pathlib import Path
import json
import os
import sys
ROOT=Path(__file__).resolve().parents[1]
DEX=ROOT/'deps/DexAgent-adapted'
sys.path.insert(0,str(DEX))
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
from tool_library.skills.search_subgoal_variants import search_subgoal_variants
out=ROOT/'continuation/runtime_probe'
probe=ROOT/'continuation/physics_probe'
plan=json.loads((out/'plan.json').read_text())
plan['subgoals'][0].pop('pose_candidates',None)
(out/'search_plan.json').write_text(json.dumps(plan,indent=2))
grasps=json.loads((probe/'grasps.json').read_text())
recon={'scope':'Nominal interaction diagnostic only. No source reconstruction certification.',
    'objects':[{'id':'test_tube','type':'rigid','root_body':'tube_tube'},
               {'id':'rack','type':'rigid','root_body':'rack_rack'}]}
(out/'nominal_body_map.json').write_text(json.dumps(recon,indent=2))
result=search_subgoal_variants(str(probe/'scene.xml'),str(out/'search_plan.json'),str(out/'search'),
    prefix_dir=str(out/'execution'),subgoal=1,knobs={'pose':{'choices':[grasps[i] for i in [0,1,2,3,6,7]]}},
    population=6,rounds=1,workers=2,max_evals=6,max_minutes=18.,eval_timeout_s=480.,dig=False,
    scene_semantic_json=str(ROOT/'reviewed_semantics/01_grasp_test_tube.json'),
    scene_recon_json=str(out/'nominal_body_map.json'))
(out/'search_result.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:result.get(k) for k in ['ok','improved','evaluations','best','next_step']}),flush=True)
