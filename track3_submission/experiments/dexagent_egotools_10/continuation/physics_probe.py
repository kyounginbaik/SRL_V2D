"""Nominal native grasp-search diagnostic, not a video reconstruction claim."""
from pathlib import Path
import json
import os
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
DEX=ROOT/'deps/DexAgent-adapted'
sys.path.insert(0,str(DEX))
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(name,'1')
os.environ.setdefault('MUJOCO_GL','egl')
from tool_library.skills.construct_round_test_tube import construct_round_test_tube
from tool_library.skills.construct_conical_tube_rack import construct_conical_tube_rack
from tool_library.skills.stage_on_bench import stage_on_bench
from tool_library.skills.generate_rigid_grasps import generate_rigid_grasps
from tool_library.verifiers.grasp_lift_test import grasp_lift_test

out=ROOT/'continuation/physics_probe'
out.mkdir(exist_ok=True)
start=time.monotonic()
tube=construct_round_test_tube(out/'tube',name='tube',radius=.006,length=.085,wall=.0006,mass=.006)
rack=construct_conical_tube_rack(out/'rack',name='rack',length=.18,width=.062,height=.055,
    rows=2,cols=6,pitch_x=.0275,pitch_y=.026,hole_r=.009,rim_r=.010,include_floor=False,
    rack_rgba=[.65,.03,.12,1.])
objects=[dict(name='rack',xml=rack['xml'],pos=[0,.35,0]),dict(name='tube',xml=tube['xml'],pos=[.04125,.337,.004])]
bench=out/'scene.xml'
staged=stage_on_bench(DEX/'setup/scene.xml',
    assets=[dict(name=x['name'],xml=x['xml'],x=x['pos'][0],y=x['pos'][1],z=x['pos'][2]) for x in objects],
    out_xml=str(bench),rig={'origin':[0,0,0],'R':[[1,0,0],[0,1,0],[0,0,1]]})
spec=dict(hand_xml=str(DEX/'setup/robot/sharpa_right.xml'),side='right',objects=objects,floor=True,
          bench_scene=str(bench))
(out/'hand_scene.json').write_text(json.dumps(spec,indent=2))
print('Starting nominal tube-in-rack grasp search',flush=True)
result=generate_rigid_grasps(spec,height_fracs=[.85,.95],yaw_step_deg=90,
    thumb_oppose_fracs=[.4],include_top=True,pinch_height_frac=[.85,.95],pinch_ik_iters=120,
    bench_scene=str(bench),out_json=str(out/'grasps.json'),verify=False)
(out/'search.json').write_text(json.dumps(result,indent=2))
print(json.dumps({'n_candidates':result.get('n_candidates'),'n_feasible':result.get('n_feasible')}),flush=True)
trials=[]
for index,pose in enumerate(result.get('grasps',[])[:4]):
    verdict=grasp_lift_test(spec,pose)
    trials.append(dict(index=index,pose=pose,verdict=verdict))
    print(json.dumps({'trial':index,'ok':verdict.get('ok'),'lifts':verdict.get('lifts'),'first_fail':verdict.get('first_fail')}),flush=True)
    (out/'trials.json').write_text(json.dumps(trials,indent=2))
    if verdict.get('ok') is True:break
(out/'diagnostic.json').write_text(json.dumps({'scope':'Nominal asset interaction diagnostic; dimensions and placement are not yet fitted to the source video.',
    'elapsed_s':time.monotonic()-start,'search':{k:v for k,v in result.items() if k!='grasps'},
    'n_trials':len(trials),'lift_test_pass':any(t['verdict'].get('ok') is True for t in trials)},indent=2))
