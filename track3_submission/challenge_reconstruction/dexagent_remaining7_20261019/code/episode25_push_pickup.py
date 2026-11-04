"""Robot-only outward push followed by underside pinch; unchanged object targets."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,itertools,argparse,multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
from episode25_pinch_transfer import W,B,E,H,G,TR,Experiment,dump,geom_pen,pinch,poses,commands,execute_reference,evaluate_files,init,evaluate

def smooth(x):
 x=np.clip(x,0,1);return x*x*(3-2*x)

def push_poses(exp,settled,out):
 m=exp.model;d=mujoco.MjData(m);rig=H.HandRig.build(m,'robot_right_','right');cols=exp.cols['right'];bid=m.body('wooden_piece_2').id;oc=exp.objects['wooden_piece_2']
 P,N,tree=G.surface(Path(exp.spec['source'])/'assets/wooden_piece_2.obj');r=Rotation.from_quat(settled[oc+3:oc+7],scalar_first=True).as_matrix();p=settled[oc:oc+3];points=P@r.T+p;normals=N@r.T
 # A real vertical board edge, outside the narrow center slot.
 mask=(normals[:,0]<-.8)&(points[:,0]<np.quantile(points[:,0],.05)+.003)
 options=[];reports=[]
 for yoffset in [-.035,.035]:
  targets={}
  for k,dy in [(1,-.012),(2,.012)]:
   ideal=np.array([points[:,0].min(),p[1]+yoffset+dy,np.median(points[:,2])]);ids=np.flatnonzero(mask);idx=ids[np.argmin(np.linalg.norm(points[ids]-ideal,axis=-1))];targets[k]=points[idx]+.001*normals[idx]
  for frame,yaw in itertools.product([145,160],[0,90,180,270]):
   q0=exp.base[125].copy();q0[cols]=exp.target[frame,cols]
   for name,a in exp.objects.items():
    if name=='wooden_piece_2':q0[a:a+7]=settled[a:a+7]
    else:q0[a:a+7]=exp.target[125,a:a+7]
   R=Rotation.from_rotvec([0,0,np.deg2rad(yaw)])*Rotation.from_euler('XYZ',q0[cols[3:6]])
   q0[cols[3:6]]=G.nearest_euler(R.as_euler('XYZ'),q0[cols[3:6]])
   d.qpos[:]=q0;mujoco.mj_forward(m,d);sites=[rig.fingertip_sites[k] for k in ['index','middle']]
   q0[cols[:3]]+=np.mean(list(targets.values()),axis=0)-d.site_xpos[sites].mean(0)
   q,ik=G.bounded_ik(m,d,rig,q0,targets,bid);d.qpos[:]=q;mujoco.mj_forward(m,d);pen=geom_pen(m,d,rig,bid)
   horizontal=0
   for c in d.contact:
    b1,b2=int(m.geom_bodyid[c.geom1]),int(m.geom_bodyid[c.geom2]);other=b2 if b1==bid else b1 if b2==bid else -1
    if other in rig.body_ids and abs(c.frame[0])>.7:horizontal+=1
   row={'frame':frame,'yaw':yaw,'y_offset':yoffset,'horizontal_contacts':horizontal,'penetration':pen,**ik};reports.append(row)
   if pen['table']<.001 and pen['target']<.004 and pen['other']<.003 and ik['max_target_residual_m']<.008 and horizontal:
    score=ik['max_target_residual_m']+pen['table']*10+pen['target'];options.append((score,q[cols].copy(),row))
 dump(out/'push_geometry.json',reports);return sorted(options,key=lambda x:x[0])[:4]

def main(a):
 out=a.output;out.mkdir(parents=True,exist_ok=True);base=W/'refinement/episode_000025';spec=json.loads((base/'peg_v1/spec.json').read_text());left=json.loads((base/'pinch_pressure_v4/result.json').read_text());spec['reference']=left['reference'];exp=Experiment(spec);dump(out/'spec.json',spec)
 settled=np.load(base/'peg_v1/passive_forecast.npz')['qpos'][-1];contacts,actual,closed,opened=pinch();pushes=push_poses(exp,settled,out)
 if not pushes:dump(out/'failed.json',{'reason':'no collision-valid robot side-push poses'});return
 print('push candidates',len(pushes),flush=True)
 task=[];combos=[];oc=exp.objects['wooden_piece_2'];cols=exp.cols['right']
 for displacement in [.05,.065]:
  shifted=settled.copy();shifted[oc]+=displacement;gout=out/f'grip_{displacement:g}';gout.mkdir(exist_ok=True);grasps=poses(exp,'right',shifted,contacts,actual,closed,opened,gout)
  for push,g in itertools.product(pushes,grasps[:3]):
   Q=exp.base.copy();right=commands(exp,'right',g,shifted,closed,opened,times=(145,160,174,182));pq=push[1]
   for t in range(1,151):
    if t<80:right[t]=right[0];continue
    if t<100:
     right[t]=right[0];right[t,2]+=.25*smooth((t-80)/20);continue
    target=pq.copy();target[0]+=displacement*smooth((t-128)/14)
    target[2]+=.12*(1-smooth((t-114)/12)) if t<128 else .12*smooth((t-142)/8)
    reach=smooth((t-100)/14);high=exp.base[0,cols].copy();high[2]+=.25;right[t]=(1-reach)*high+reach*target
   for t in range(151,161):
    blend=smooth((t-150)/10);right[t]=(1-blend)*right[150]+blend*right[t]
   Q[:,cols]=np.clip(right,exp.limits['right'][:,0],exp.limits['right'][:,1]);Q[0]=exp.target[0]
   for adr in exp.objects.values():assert np.array_equal(Q[:,adr:adr+7],exp.target[:,adr:adr+7])
   i=len(task);p=out/'proposals'/f'{i:04d}.npz';p.parent.mkdir(exist_ok=True);TR.save_reference(p,exp.model,Q,exp.ref['labels'],exp.fps,exp.ref['segment'],exp.ref['segments'],source='Physical robot side-push then underside pinch; objects never actuated or repositioned',arms_coupled=False);task.append((i,str(p)));combos.append({'planned_outward_push_m':displacement,'push':push[2],'grasp':g[-1]})
 dump(out/'combinations.json',combos);reports=[]
 if not task:dump(out/'failed.json',{'reason':'no underside grasps after planned push'});return
 with ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('spawn'),initializer=init,initargs=(spec,)) as pool:
  for index,r in pool.map(evaluate,task):
   row={'index':index,**r};reports.append(row);dump(out/'progress.json',{'count':len(reports),'total':len(task),'best':min(reports,key=lambda x:x['cost'])});print(json.dumps(row),flush=True)
 dump(out/'scores.json',reports);best=min(reports,key=lambda x:x['cost']);path=task[best['index']][1];execution=execute_reference(spec['scene'],path,out/'execution',expected_duration_s=(len(exp.base)-1)/exp.fps);manifest=json.loads((E/'input_manifest.json').read_text());metrics=evaluate_files(manifest['source'],execution['trajectory'],E/'input_manifest.json',execution['scene'],E/'metric_geometry.json');dump(out/'result.json',{'episode':'episode_000025','reference':path,'best':best,'execution':execution,'metrics':metrics,'learned_policy':False,'object_targets_unchanged':True,'initial_state_unchanged':True})

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=8);main(p.parse_args())
