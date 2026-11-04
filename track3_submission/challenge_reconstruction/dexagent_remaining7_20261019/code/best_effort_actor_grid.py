#!/usr/bin/env python3
"""Best-effort actor fit: like spoon_actor_grid, but the teacher and the actor need not pass the proxy MP-SR. Selection is by lowest maximum error ratio among physically verified replays. Output is explicitly flagged."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
BASE=Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019')
LEGACY=BASE.parent/'dexagent_final_recon_20261017'
sys.path.insert(0,str(LEGACY/'code'))
sys.path.insert(0,'/data1/SRL_V2D/third_party/DexAgent')

def write(p,r):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(r,indent=2)+'\n')

def attempt(params):
 episode,source,output,centers,gain,velocity,epdir,seed=params
 from learned_feedback_policy import train
 from evaluate_policy_component import evaluate
 source,output=Path(source),Path(output)
 r=json.loads((source/'result.json').read_text())
 assert r["execution"]["physics_verified"]  # best-effort: teacher need not pass the proxy metric
 dest=output/f'actor_c{centers}_p{gain:g}_v{velocity:g}'
 checkpoint=dest/'training/policy.npz'
 if not checkpoint.exists():train(r['execution']['scene'],r['execution']['trajectory'],dest/'training',centers=centers,feedback_gain=gain,velocity_gain=velocity,seed=seed)
 epdir=Path(epdir)
 if not (dest/'evaluation/result.json').exists():evaluate(episode,r['execution']['scene'],checkpoint,dest/'evaluation',epdir)
 ev=json.loads((dest/'evaluation/result.json').read_text());obs=ev['metrics']['official_style_reference_proxy']['objects']
 return dict(episode=f'episode_{episode:06d}',teacher=str(source),checkpoint=str(checkpoint),training_report=str(dest/'training/training_report.json'),evaluation=str(dest/'evaluation'),episode_dir=str(epdir),passed=ev['execution']['physics_verified'] and ev['execution'].get('independent_control_replay',{}).get('ok',False),proxy_mp_sr=ev['metrics']['reference_proxy_mp_sr'],object_errors={k:dict(cm=v['mean_position_error_cm'][0],deg=v['mean_orientation_error_degrees'][0]) for k,v in obs.items()},maximum_ratio=max(max(v['mean_position_error_cm'][0]/3,v['mean_orientation_error_degrees'][0]/30) for v in obs.values()),physical_replay_verified=ev['execution']['physics_verified'],hand_source='HaWoR',submission_ready=False,portable_validation_pending=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--episode',type=int,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--episode-dir',type=Path,required=True);p.add_argument('--seed',type=int,default=44);a=p.parse_args()
 grid=[(1024,.001,.00005),(1024,.003,.0001),(1024,.01,.0003),(1024,.03,.0005),(1024,.10,.0005),(768,.003,.0001),(1536,.003,.0001),(1024,.003,.0003)]
 args=[(a.episode,str(a.source),str(a.output),*g,str(a.episode_dir),a.seed) for g in grid]
 with ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('spawn')) as pool:
  reports=[]
  for row in pool.map(attempt,args):
   reports.append(row);write(a.output/'attempts.json',reports);print(json.dumps(row),flush=True)
 passed=[r for r in reports if r['passed']]
 if passed:
  best=min(passed,key=lambda r:r['maximum_ratio']);best['reference_proxy_mp_sr']=best['proxy_mp_sr'];best['best_effort_not_passing']=best['proxy_mp_sr']!=1.;write(a.output/'completed.json',best)
 else:write(a.output/'failed.json',dict(attempts=len(reports),all_failed=True,submission_ready=False))
if __name__=='__main__':main()
