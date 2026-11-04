"""Mounted-arm diagnostic on nominal assets; not a source reconstruction result."""
from pathlib import Path
import json
import os
import sys
ROOT=Path(__file__).resolve().parents[1]
DEX=ROOT/'deps/DexAgent-adapted'
sys.path.insert(0,str(DEX))
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
os.environ.setdefault('MUJOCO_GL','egl')
from tool_library.skills.execute_subgoal_plan import execute_subgoal_plan
from tool_library.skills.render_trajectory_video import render_trajectory_video
probe=ROOT/'continuation/physics_probe'
out=ROOT/'continuation/runtime_probe'
out.mkdir(exist_ok=True)
search=json.loads((probe/'grasps.json').read_text())
grasps=search['grasps'] if isinstance(search,dict) else search
plan={'subgoals':[
    dict(index=1,motion='grasp_lift',side='right',object='tube_tube',pose=grasps[0],
         pose_candidates=grasps,hand_scene=str(probe/'hand_scene.json'),lift_m=.10,
         expect={'untouched':['rack_rack']}),
    dict(index=2,motion='hold',side='right',object='tube_tube',duration_s=1.3)]}
(out/'plan.json').write_text(json.dumps(plan,indent=2))
print('Executing nominal candidate with the fixed-tree YAM arm and bounded motors.',flush=True)
result=execute_subgoal_plan(str(probe/'scene.xml'),plan,str(out/'execution'))
(out/'result.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
files=result.get('files',{})
if files.get('trajectory') and Path(files['trajectory']).exists():
    rendered=render_trajectory_video(files['scene_physics'],files['trajectory'],str(out/'video'),cameras=['ego'],stride=2,width=640,height=480)
    (out/'render.json').write_text(json.dumps(rendered,indent=2))
