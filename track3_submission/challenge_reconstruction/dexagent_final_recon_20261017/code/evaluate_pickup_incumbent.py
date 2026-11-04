"""Snapshot an active pickup search incumbent and independently test full horizon."""
from pathlib import Path
import argparse,json,numpy as np
from physics_pickup_search import PickupExperiment,dump
from floating_runtime import execute_reference
from tool_library.common import tracking as TR
from evaluate_mp_sr import evaluate_files

p=argparse.ArgumentParser();p.add_argument('--search',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
settings=json.loads((a.search/'settings.json').read_text());progress=json.loads((a.search/'progress.json').read_text());spec=settings['spec']
if (a.output/'result.json').exists():raise FileExistsError(a.output/'result.json')
a.output.mkdir(parents=True,exist_ok=True);dump(a.output/'search_snapshot.json',progress)
exp=PickupExperiment(spec);x=np.asarray(progress['best_parameters']);Q,timing=exp.commands(x)
unchanged=np.setdiff1d(np.arange(exp.m.nq),exp.cols)
assert np.array_equal(Q[:,unchanged],exp.Q[:,unchanged]);assert np.array_equal(Q[:exp.start_frame+1],exp.Q[:exp.start_frame+1])
path=TR.save_reference(a.output/'reference.npz',exp.m,Q,exp.ref['labels'],exp.fps,exp.ref['segment'],exp.ref['segments'],source='Frozen physical pickup search incumbent; full-horizon independent replay',arms_coupled=False,floating_runtime=True)
execution=execute_reference(spec['scene'],path,a.output/'execution')
metrics=evaluate_files(spec['source'],execution['trajectory'],spec['manifest'],execution['scene'],spec['geometry'])
result=dict(search=str(a.search),snapshot_iteration=progress['history'][-1]['iteration'],parameters=x,reference=str(path),execution=execution,metrics=metrics)
dump(a.output/'result.json',result)
print(json.dumps({'output':str(a.output),'physics_verified':execution['physics_verified'],'metrics':metrics['declared_frame_reference_proxy']},indent=2))
