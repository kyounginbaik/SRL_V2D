"""Physically forecast a robot push before planning a side-entry board pinch."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,itertools,multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from episode25_pinch_transfer import W,B,E,H,G,TR,Experiment,dump,pinch,pinch_pressure,poses,commands,execute_reference,evaluate_files,init,evaluate,servo_rollout
from episode25_push_pickup import push_poses,smooth

def forecast(task):
 spec,Q,index=task;exp=Experiment(spec);r=servo_rollout(exp.model,Q[:151],exp.fps,qpos_start=exp.initial,record_substeps=False)
 return index,r

def push_commands(exp,pq,distance):
 cols=exp.cols['right'];Q=exp.base.copy();right=Q[:,cols].copy();right[:]=right[0]
 for t in range(80,151):
  if t<100:right[t]=right[0];right[t,2]+=.25*smooth((t-80)/20);continue
  target=pq.copy();target[0]+=distance*smooth((t-128)/14)
  target[2]+=.12*(1-smooth((t-114)/12)) if t<128 else .12*smooth((t-142)/8)
  reach=smooth((t-100)/14);high=exp.base[0,cols].copy();high[2]+=.25;right[t]=(1-reach)*high+reach*target
 Q[:,cols]=right;Q[0]=exp.target[0];return Q

def main(a):
 out=a.output;out.mkdir(parents=True,exist_ok=True);base=W/'refinement/episode_000025';spec=json.loads((base/'peg_v1/spec.json').read_text());spec['reference']=json.loads((base/'pinch_pressure_v4/result.json').read_text())['reference'];exp=Experiment(spec);dump(out/'spec.json',spec)
 settled=np.load(base/'peg_v1/passive_forecast.npz')['qpos'][-1];contacts,actual,closed,opened=pinch();pushes=push_poses(exp,settled,out);prefix=[];meta=[]
 for d,push in itertools.product([.065,.08,.095],pushes):
  Q=push_commands(exp,push[1],d);prefix.append((spec,Q,len(prefix)));meta.append({'commanded_push_m':d,'push_geometry':push[2]})
 cols=exp.cols['right'];oc=exp.objects['wooden_piece_2'];tasks=[];combos=[];forecasts=[]
 with ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('spawn')) as pool:
  for index,r in pool.map(forecast,prefix):
   np.savez_compressed(out/f'push_forecast_{index:02d}.npz',**{k:v for k,v in r.items() if isinstance(v,np.ndarray)});physical=r['qpos'][-1].copy();shift=physical[oc:oc+3]-settled[oc:oc+3];row={**meta[index],'actual_shift_m':shift.tolist(),'actual_pose':physical[oc:oc+7].tolist()};forecasts.append(row);print('push',index,row,flush=True)
   if r['unstable'] or abs(physical[oc+2])>.05 or np.linalg.norm(shift)>.15:continue
   gout=out/f'grip_{index:02d}';gout.mkdir(exist_ok=True);grasps=poses(exp,'right',physical,contacts,actual,closed,opened,gout)
   for g,pressure,approach in itertools.product(grasps[:6],[.005,.008,.012],['radial']):
    Q=prefix[index][1].copy();right=commands(exp,'right',g,physical,pinch_pressure(closed,pressure),opened,times=(151,160,174,183))
    # The grasp can be below the tabletop. Reach it horizontally from outside;
    # descending from a vertical hover would collide with the board itself.
    radial=physical[oc:oc+2]-np.array([.0441665,.307367]);radial/=np.linalg.norm(radial)
    target=right[174].copy();pre=target.copy();pre[:2]+=.16*radial;pre[6:]=opened
    high=pre.copy();high[2]=max(.22,pre[2]+.2)
    for t in range(151,161):
     f=smooth((t-150)/10);right[t]=(1-f)*Q[150,cols]+f*high
    for t in range(161,169):
     f=smooth((t-160)/8);right[t]=(1-f)*high+f*pre
    for t in range(169,180):
     f=smooth((t-168)/11);right[t]=(1-f)*pre+f*target;right[t,6:]=opened
    for t in range(180,190):
     f=smooth((t-179)/10);right[t]=target;right[t,6:]=(1-f)*opened+f*pinch_pressure(closed,pressure)
    # Retain actual object pose until closure, then join the original schedule.
    held=commands(exp,'right',g,physical,pinch_pressure(closed,pressure),opened,times=(151,160,180,189))
    right[190:]=held[190:];Q[151:,cols]=np.clip(right[151:],exp.limits['right'][:,0],exp.limits['right'][:,1]);Q[0]=exp.target[0]
    for adr in exp.objects.values():assert np.array_equal(Q[:,adr:adr+7],exp.target[:,adr:adr+7])
    i=len(tasks);p=out/'proposals'/f'{i:04d}.npz';p.parent.mkdir(exist_ok=True);TR.save_reference(p,exp.model,Q,exp.ref['labels'],exp.fps,exp.ref['segment'],exp.ref['segments'],source='HaWoR robot refinement with physically forecast outward push and horizontal underside pinch',arms_coupled=False);tasks.append((i,str(p)));combos.append({'push_index':index,'pressure_m':pressure,'grasp':g[-1],'approach':approach})
 dump(out/'forecasts.json',forecasts);dump(out/'combinations.json',combos)
 if not tasks:dump(out/'failed.json',{'reason':'No feasible grips after physical push'});return
 rows=[]
 with ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('spawn'),initializer=init,initargs=(spec,)) as pool:
  for index,r in pool.map(evaluate,tasks):
   row={'index':index,**r};rows.append(row);dump(out/'progress.json',{'count':len(rows),'total':len(tasks),'best':min(rows,key=lambda x:x['cost'])});print(json.dumps(row),flush=True)
 dump(out/'scores.json',rows);best=min(rows,key=lambda x:x['cost']);path=tasks[best['index']][1];execution=execute_reference(spec['scene'],path,out/'execution',expected_duration_s=(len(exp.base)-1)/exp.fps);manifest=json.loads((E/'input_manifest.json').read_text());metrics=evaluate_files(manifest['source'],execution['trajectory'],E/'input_manifest.json',execution['scene'],E/'metric_geometry.json');dump(out/'result.json',{'episode':'episode_000025','reference':path,'best':best,'execution':execution,'metrics':metrics,'learned_policy':False,'object_targets_unchanged':True,'initial_state_unchanged':True})

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=8);main(p.parse_args())
