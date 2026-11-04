#!/usr/bin/env python3
"""Episode 36 actor grid: fit the registered learned actor to a roll re-plan teacher, evaluate with the unchanged policy
evaluator against the corrected-track episode dir."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
LEG=Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017'); EPDIR=LEG/'refinement_20261021_ep36/episode_corrected/episode_000036'
sys.path.insert(0,str(LEG/'code'));sys.path.insert(0,'/data1/SRL_V2D/third_party/DexAgent')
def attempt(p):
 source,output,centers,gain,velocity=p
 from learned_feedback_policy import train
 from evaluate_policy_component import evaluate
 source,output=Path(source),Path(output); r=json.loads((source/'result.json').read_text())
 dest=output/f'actor_c{centers}_p{gain:g}_v{velocity:g}'; ck=dest/'training/policy.npz'
 if not ck.exists():train(r['execution']['scene'],r['execution']['trajectory'],dest/'training',centers=centers,feedback_gain=gain,velocity_gain=velocity,seed=44)
 if not (dest/'evaluation/result.json').exists():evaluate(36,r['execution']['scene'],ck,dest/'evaluation',EPDIR)
 ev=json.loads((dest/'evaluation/result.json').read_text());o=ev['metrics']['official_style_reference_proxy']['objects']
 return dict(teacher=str(source),checkpoint=str(ck),training_report=str(dest/'training/training_report.json'),evaluation=str(dest/'evaluation'),episode_dir=str(EPDIR),passed=ev['metrics']['reference_proxy_mp_sr']==1. and ev['execution']['physics_verified'],object_errors={k:dict(cm=v['mean_position_error_cm'][0],deg=v['mean_orientation_error_degrees'][0]) for k,v in o.items()},maximum_ratio=max(max(v['mean_position_error_cm'][0]/3,v['mean_orientation_error_degrees'][0]/30) for v in o.values()))
def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=8);a=p.parse_args()
 grid=[(512,.03,.0005),(1024,.03,.0005),(512,.01,.0002),(1024,.01,.0002),(512,.003,.0001),(1024,.003,.0001),(2048,.003,.0001),(2048,.01,.0002)]
 with ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('spawn')) as pool:
  rows=[]
  for r in pool.map(attempt,[(str(a.source),str(a.output),*g) for g in grid]):
   rows.append(r);(a.output/'attempts.json').write_text(json.dumps(rows,indent=2));print(json.dumps(dict(ck=r['checkpoint'].split('/')[-3],passed=r['passed'],err=r['object_errors'])),flush=True)
 ok=[r for r in rows if r['passed']]
 (a.output/('completed.json' if ok else 'failed.json')).write_text(json.dumps(min(ok,key=lambda r:r['maximum_ratio']) if ok else dict(all_failed=True),indent=2))
if __name__=='__main__':main()
