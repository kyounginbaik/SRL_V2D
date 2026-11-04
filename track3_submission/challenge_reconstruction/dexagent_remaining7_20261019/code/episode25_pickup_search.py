#!/usr/bin/env python3
"""Search side-entry approaches; short diagnostics never certify submission."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,json,itertools,sys,multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from refine_episode25_geometry import B,W,E,Experiment,TR,H,G,servo_rollout,execute_reference,evaluate_files,angular_errors,dump

def smooth(x):
    x=np.clip(x,0,1);return x*x*(3-2*x)

def candidate(exp,seed,side,press,zshift,open_amount,distance):
    Q=exp.base.copy();fps=exp.fps;cols=exp.cols[side];oc=exp.objects['wooden_piece_1' if side=='left' else 'wooden_piece_2']
    rig=H.HandRig.build(exp.model,f'robot_{side}_',side)
    close=48 if side=='left' else 174;begin=12 if side=='left' else 130;pre=30 if side=='left' else 150;arrive=42 if side=='left' else 166
    anchor=60 if side=='left' else 190
    R=Rotation.from_quat(Q[anchor,oc+3:oc+7],scalar_first=True).as_matrix()
    localp=R.T@(seed[anchor,cols[:3]]-Q[anchor,oc:oc+3]);localR=R.T@H.euler_to_mat(seed[anchor,cols[3:6]])
    finger=seed[anchor,cols[6:]].copy();opened=finger.copy()
    for j,name in enumerate(rig.finger_qpos):
        if any(k in name for k in ['MCP_FE','PIP','DIP','thumb_IP']):finger[j]+=press;opened[j]-=open_amount
    passive=np.load(W/'refinement/episode_000025/geometry_v3/passive_forecast.npz')['qpos']
    settled=passive[-1,oc:oc+7];settledR=Rotation.from_quat(settled[3:],scalar_first=True)
    previous=Q[0,cols[3:6]].copy();direction=np.array([0,0,1.]) if side=='left' else np.array([-1.,0,0])
    for t in range(1,len(Q)):
        if t<begin:
            if side=='right':Q[t,cols]=Q[0,cols]
            continue
        fade=smooth((t-close)/8.);goal=Rotation.from_quat(exp.base[t,oc+3:oc+7],scalar_first=True)
        oR=(settledR*Rotation.from_rotvec((settledR.inv()*goal).as_rotvec()*fade)).as_matrix()
        op=(1-fade)*settled[:3]+fade*exp.base[t,oc:oc+3]
        reach=smooth((t-pre)/(arrive-pre));opening=smooth((t-arrive)/(close-arrive))
        pos=op+oR@(localp+direction*distance*(1-reach));pos[2]+=zshift
        rot=G.nearest_euler(H.mat_to_euler(oR@localR),previous);previous=rot
        fingers=(1-opening)*opened+opening*finger
        target=np.r_[pos,rot,fingers]
        blend=smooth((t-begin)/(pre-begin))
        Q[t,cols]=(1-blend)*exp.base[t,cols]+blend*target
    Q[:,cols]=np.clip(Q[:,cols],exp.limits[side][:,0],exp.limits[side][:,1]);Q[0]=exp.base[0]
    if side=='left':
        # Avoid the erroneous transient second-hand excursion during pickup.
        c=exp.cols['right']
        for t in range(1,145):Q[t,c]=Q[0,c]
    return Q

_EXP=None;_SIDE=None;_END=None
def init(spec,side,end):
    global _EXP,_SIDE,_END
    _EXP=Experiment(spec);_SIDE=side;_END=end

def worker(task):
    index,path=task;exp=_EXP;Q=np.load(path)['qpos'][:_END]
    r=servo_rollout(exp.model,Q,exp.fps,qpos_start=exp.initial,record_substeps=False)
    if r['unstable']:return {'index':index,'cost':1e9,'unstable':True}
    p,q=exp.object_poses(r['qpos']);oid=0 if _SIDE=='left' else 1
    pe=np.linalg.norm(p-exp.tp[:len(p)],axis=-1);ae=angular_errors(q[:,None],exp.tq[:len(p),None],exp.axes,exp.kinds)[:,0]
    start=48 if oid==0 else 174;losspe=pe[start:,oid].mean();lossae=ae[start:,oid].mean()
    cost=losspe/.03+lossae/np.deg2rad(30)
    return {'index':index,'cost':float(cost),'mean_position_cm':(pe.mean(0)*100).tolist(),'mean_orientation_deg':np.rad2deg(ae.mean(0)).tolist(),
        'carry_position_cm':float(losspe*100),'carry_orientation_deg':float(np.rad2deg(lossae)),
        'object_max_z':r['qpos'][:,exp.objects[exp.names[oid]]+2].max().item(),
        'max_hand_pen_m':float(np.max(r['hand_pen_m']))}

def main(a):
    out=a.output;out.mkdir(parents=True,exist_ok=True)
    spec=json.loads((W/'refinement/episode_000025/geometry_v3/spec.json').read_text())
    if a.reference:spec['reference']=str(a.reference)
    exp=Experiment(spec);dump(out/'spec.json',spec)
    seedids=[0,36,72,108] if a.side=='left' else [0,3,6,9]
    seeds=[np.load(W/'refinement/episode_000025/geometry_v3/proposals'/f'{i:04d}.npz')['qpos'] for i in seedids]
    tasks=[];params=[]
    for i,(si,press,z,opening,distance) in enumerate(itertools.product(range(4),[0.,.15,.3],[-.003,.003],[.1,.3],[.035,.055])):
        Q=candidate(exp,seeds[si],a.side,press,z,opening,distance)
        path=out/'proposals'/f'{i:04d}.npz';path.parent.mkdir(exist_ok=True)
        TR.save_reference(path,exp.model,Q,exp.ref['labels'],exp.fps,exp.ref['segment'],exp.ref['segments'],source='HaWoR-seeded side-entry board pickup',arms_coupled=False)
        tasks.append((i,str(path)));params.append({'seed':seedids[si],'press':press,'z_shift':z,'opening':opening,'side_entry_distance':distance})
    dump(out/'parameters.json',params);reports=[]
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('spawn'),initializer=init,initargs=(spec,a.side,a.frames)) as pool:
        for row in pool.map(worker,tasks):
            reports.append(row);dump(out/'progress.json',{'evaluated':len(reports),'total':len(tasks),'best':min(reports,key=lambda x:x['cost'])});print(json.dumps(row),flush=True)
    dump(out/'scores.json',reports);best=min(reports,key=lambda x:x['cost']);path=Path(tasks[best['index']][1])
    execution=execute_reference(spec['scene'],path,out/'execution',expected_duration_s=(len(exp.base)-1)/exp.fps)
    manifest=json.loads((E/'input_manifest.json').read_text());metrics=evaluate_files(manifest['source'],execution['trajectory'],E/'input_manifest.json',execution['scene'],E/'metric_geometry.json')
    dump(out/'result.json',{'episode':'episode_000025','reference':str(path),'best':best,'execution':execution,'metrics':metrics,'learned_policy':False,'search_scope':f'first {a.frames} frames {a.side} diagnostic; final metrics use entire episode'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=8);p.add_argument('--side',choices=['left','right'],default='left');p.add_argument('--frames',type=int,default=150);p.add_argument('--reference',type=Path);main(p.parse_args())
