#!/usr/bin/env python3
"""Distil a physical teacher (dir with result.json + execution/{scene_physics.xml,trajectory.npz}) into the registered RBF actor at
several hyperparameters and evaluate each with the unchanged evaluate_policy_component in the teacher's own frozen scene against an
explicit episode dir. Stream C copy of rackscan_20261021/code/actor_grid.py with explicit --episode-dir and --grid.
usage: actor_grid_c.py --episode E --teacher DIR --output DIR --episode-dir DIR [--grid c,p,v ...] [--workers N] [--seed 44]"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import argparse, json, sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
LEGACY=Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017')
sys.path.insert(0,str(LEGACY/'code')); sys.path.insert(0,'/data1/SRL_V2D/third_party/DexAgent')
def write(p,r): p=Path(p); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(r,indent=2)+'\n')
def attempt(params):
    episode,source,output,epdir,centers,gain,velocity,seed=params
    from learned_feedback_policy import train
    from evaluate_policy_component import evaluate
    source,output=Path(source),Path(output); r=json.loads((source/'result.json').read_text())
    assert r['execution']['physics_verified']
    dest=output/f'actor_c{centers}_p{gain:g}_v{velocity:g}'; ck=dest/'training/policy.npz'
    try:
        if not ck.exists(): train(r['execution']['scene'],r['execution']['trajectory'],dest/'training',centers=centers,feedback_gain=gain,velocity_gain=velocity,seed=seed)
        if not (dest/'evaluation/result.json').exists(): evaluate(episode,r['execution']['scene'],str(ck),str(dest/'evaluation'),epdir)
    except Exception as e:
        return dict(episode=f'episode_{episode:06d}',checkpoint=str(ck),passed=False,error=repr(e)[:300])
    ev=json.loads((dest/'evaluation/result.json').read_text()); obs=ev['metrics']['official_style_reference_proxy']['objects']
    return dict(episode=f'episode_{episode:06d}',teacher=str(source),checkpoint=str(ck),evaluation=str(dest/'evaluation'),episode_dir=str(epdir),
        passed=ev['metrics']['reference_proxy_mp_sr']==1. and ev['execution']['physics_verified'] and ev['execution'].get('independent_control_replay',{}).get('ok',False),
        object_errors={k:dict(cm=round(v['mean_position_error_cm'][0],3),deg=round(v['mean_orientation_error_degrees'][0],2)) for k,v in obs.items()},
        maximum_ratio=max(max(v['mean_position_error_cm'][0]/3,v['mean_orientation_error_degrees'][0]/30) for v in obs.values()))
def main():
    p=argparse.ArgumentParser(); p.add_argument('--episode',type=int,required=True); p.add_argument('--teacher',required=True); p.add_argument('--output',required=True)
    p.add_argument('--episode-dir',required=True); p.add_argument('--grid',nargs='+',default=['512,0.001,0.00005','1024,0.001,0.00005','2048,0.001,0.00005','512,0.003,0.0001','1024,0.003,0.0001','2048,0.003,0.0001','1024,0.01,0.0005','1024,0.03,0.0005'])
    p.add_argument('--workers',type=int,default=4); p.add_argument('--seed',type=int,default=44); a=p.parse_args()
    grid=[tuple(float(x) if i else int(x) for i,x in enumerate(g.split(','))) for g in a.grid]
    args=[(a.episode,a.teacher,a.output,a.episode_dir,*g,a.seed) for g in grid]; reports=[]
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('spawn')) as pool:
        for row in pool.map(attempt,args): reports.append(row); write(Path(a.output)/'attempts.json',reports); print(json.dumps(row),flush=True)
    passed=[r for r in reports if r['passed']]
    if passed: best=min(passed,key=lambda r:r['maximum_ratio']); write(Path(a.output)/'completed.json',best); print('BEST',json.dumps(best),flush=True)
    else: write(Path(a.output)/'failed.json',dict(attempts=len(reports),all_failed=True))
if __name__=='__main__': main()
