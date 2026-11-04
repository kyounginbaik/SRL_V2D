"""Rebuild a pickup-search teacher from its best parameters with modified timing (e.g. a later release so the second hand opens
only once the spoon rests), execute full-horizon, score. usage: rebuild_pickup.py RESULT_DIR OUT_DIR --release 19.5 [--horizon 8]"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import argparse, json, sys
from pathlib import Path
import numpy as np
R=Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017'); sys.path[:0]=[str(R/'code'),'/data1/SRL_V2D/third_party/DexAgent']
from physics_pickup_search_v2 import PickupExperiment, dump
from tool_library.common import tracking as TR
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files
p=argparse.ArgumentParser(); p.add_argument('src'); p.add_argument('out'); p.add_argument('--release',type=float); p.add_argument('--horizon',type=float)
a=p.parse_args(); src=Path(a.src); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
r=json.loads((src/'result.json').read_text()); spec=dict(r['spec']); spec['times']=dict(spec['times'])
if a.release is not None: spec['times']['release']=a.release
if a.horizon is not None: spec['horizon_s']=a.horizon
exp=PickupExperiment(spec); best=np.asarray(r['parameters'],float); Q,timing=exp.commands(best)
refp=TR.save_reference(out/'reference.npz',exp.m,Q,exp.ref['labels'],exp.fps,exp.ref['segment'],exp.ref['segments'],source=f'pickup search best parameters of {src} rebuilt with times {spec["times"]}; immutable object targets/clock',arms_coupled=False,floating_runtime=True)
ex=execute_reference(spec['scene'],refp,out/'execution'); met=evaluate_files(spec['source'],ex['trajectory'],spec['manifest'],ex['scene'],spec['geometry'])
dump(out/'result.json',dict(episode=Path(spec['manifest']).parent.name,input_dir=str(Path(spec['manifest']).parent),reference=str(refp),execution=ex,metrics=met,parameters=best,spec=spec,timing=timing,source=str(src),learned_policy=False))
o=met['official_style_reference_proxy']['objects']; print(json.dumps(dict(out=str(out),mp_sr=met['reference_proxy_mp_sr'],objects={k:(round(v['mean_position_error_cm'][0],2),round(v['mean_orientation_error_degrees'][0],1)) for k,v in o.items()},completed=ex['execution_completed'])),flush=True)
