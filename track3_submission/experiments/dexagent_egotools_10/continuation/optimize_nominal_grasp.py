"""Search contact offsets and orientation after the nominal mounted-arm failures."""
from pathlib import Path
import os
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'deps/DexAgent-adapted'))
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
from tool_library.skills.optimize_grasp_pose import optimize_grasp_pose
from tool_library.skills.execute_subgoal_plan import execute_subgoal_plan
probe=ROOT/'continuation/physics_probe';out=ROOT/'continuation/runtime_probe'
grasps=json.loads((probe/'grasps.json').read_text())
r=optimize_grasp_pose(str(probe/'hand_scene.json'),family='seeds',starts=grasps[:3],
    tests=['grasp_lift_test','grasp_hold_test'],bench_scene=str(probe/'scene.xml'),
    arm_waypoints=[[0,0,0],[0,0,.1]],iterations=2,population=6,max_evals=16,max_starts=3,
    max_overlap_m=.003,workers=2,out_json=str(out/'optimized_pose.json'),
    checkpoint_json=str(out/'grasp_optimization_checkpoint.json'))
(out/'optimization_result.json').write_text(json.dumps(r,indent=2))
print(json.dumps({k:r.get(k) for k in ['ok','status','n_evals','n_tested','failure_classes']}),flush=True)
if r.get('ok') and r.get('pose'):
    plan=json.loads((out/'plan.json').read_text())
    plan['subgoals'][0]['pose']=r['pose']
    plan['subgoals'][0].pop('pose_candidates',None)
    (out/'plan_optimized.json').write_text(json.dumps(plan,indent=2))
    executed=execute_subgoal_plan(str(probe/'scene.xml'),plan,str(out/'execution_optimized'))
    (out/'result_optimized.json').write_text(json.dumps(executed,indent=2))
    print(json.dumps({'executed_ok':executed.get('ok'),'first_failure':executed.get('first_failure')}),flush=True)
