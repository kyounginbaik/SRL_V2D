#!/usr/bin/env python3
"""Retain a physically acquired pot grasp; object targets and initial state fixed."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json
from pathlib import Path
import numpy as np
from pot_pinch_search import PotPinchSearch
from prepare_seven import write
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files
from tool_library.common import tracking as TR

def run(a):
 r=json.loads((a.source/'result.json').read_text());s=r['spec'].copy();old=PotPinchSearch(s);oldQ,_=old.commands(np.asarray(r['parameters']))
 s['times']=s['times'].copy();s['times']['release']=(len(oldQ)-1)/old.fps+.1
 exp=PotPinchSearch(s);Q,_=exp.commands(np.asarray(r['parameters']))
 if a.output.exists():raise FileExistsError(a.output)
 a.output.mkdir(parents=True);assert np.array_equal(Q[:int(old.ts['release']*old.fps)],oldQ[:int(old.ts['release']*old.fps)])
 for c in exp.objects.values():assert np.array_equal(Q[:,c:c+7],exp.original[:,c:c+7])
 path=TR.save_reference(a.output/'reference.npz',exp.m,Q,exp.ref['labels'],exp.fps,exp.ref['segment'],exp.ref['segments'],source='Retain measured physical robot grasp through original episode end; unchanged object targets and home',floating_runtime=True,no_retiming=True)
 ex=execute_reference(s['scene'],path,a.output/'execution');metrics=evaluate_files(s['source'],ex['trajectory'],s['manifest'],ex['scene'],s['geometry'])
 write(a.output/'result.json',dict(episode='episode_000028',reference=str(path),execution=ex,metrics=metrics,spec=s,source=str(a.source),learned_policy=False));print(json.dumps(dict(output=str(a.output),mp_sr=metrics['reference_proxy_mp_sr'],objects=metrics['official_style_reference_proxy']['objects'])),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
