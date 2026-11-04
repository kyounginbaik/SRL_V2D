#!/usr/bin/env python3
"""Physics-tested opposing grasps for the two boards in episode 25.

Uses HaWoR-derived wrist/finger poses as geometric seeds. Only robot commands
change. The real scene home, object targets, physics and source timing are fixed.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='1'
import argparse,json,sys,itertools,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

ROOT=Path('/data1/SRL_V2D')
B=ROOT/'challenge_reconstruction/dexagent_remaining7_20261019'
L=ROOT/'challenge_reconstruction/dexagent_final_recon_20261017'
sys.path[:0]=[str(ROOT/'third_party/DexAgent'),str(L/'code')]
from tool_library.common import tracking as TR, hand as H
import grasp_geometry_seeds_v8 as G
from floating_runtime import servo_rollout,execute_reference,_freeze
from evaluate_mp_sr import evaluate_files,angular_errors
from refine_grasp_cem import Experiment,dump

W=Path(os.environ.get('E25_WORKSPACE',str(B/'physics_collision_v2')))
E=W/'episodes/episode_000025'
SIDE_SPEC={'left':('wooden_piece_1',[38,42,46,50],34,46,355),
           'right':('wooden_piece_2',[154,160,166,172],148,170,355)}

def geom_pen(model,data,rig,target):
    p={'target':0.,'table':0.,'other':0.,'self':0.}
    for c in data.contact:
        b1,b2=int(model.geom_bodyid[c.geom1]),int(model.geom_bodyid[c.geom2])
        h1,h2=b1 in rig.body_ids,b2 in rig.body_ids
        if not(h1 or h2):continue
        if h1 and h2:k='self'
        else:
            b=b2 if h1 else b1
            k='target' if b==target else 'table' if model.body(b).name=='table' else 'other'
        p[k]=max(p[k],-float(c.dist))
    return p

def board_targets(tips,points,normals):
    """Broad opposing board faces when a local neighbor query misses the back."""
    best=None
    for finger in [1,2,3]:
        xz=(tips[0,[0,2]]+tips[finger,[0,2]])/2
        xz=np.clip(xz,[-.09,-.05],[.09,.05])
        for sign in [1.,-1.]:
            a=np.flatnonzero(normals[:,1]*sign>.85)
            b=np.flatnonzero(normals[:,1]*sign<-.85)
            if not len(a) or not len(b):continue
            ia=a[np.argmin(np.linalg.norm(points[a][:,[0,2]]-xz,axis=1))]
            ib=b[np.argmin(np.linalg.norm(points[b][:,[0,2]]-xz,axis=1))]
            pa,pb=points[ia],points[ib];v=pa-pb;sep=np.linalg.norm(v)
            if sep<.002 or sep>.05 or normals[ia]@v<.7*sep or normals[ib]@(-v)<.7*sep:continue
            score=np.linalg.norm(pa-tips[0])**2+np.linalg.norm(pb-tips[finger])**2
            if best is None or score<best[0]:best=(score,finger,ia,ib)
    if best is None:return None
    _,finger,ia,ib=best
    return {0:points[ia]+.0015*normals[ia],finger:points[ib]+.0015*normals[ib]}, {'opposing_fingers':['thumb',G.FINGERS[finger]],'normal_dot':float(normals[ia]@normals[ib]),'contact_separation_m':float(np.linalg.norm(points[ia]-points[ib])),'proposal':'global opposing board faces'}

def peg_targets(tips,points,normals):
    # Raised fiducial supports are part of the exact rigid object mesh. Pinching
    # their vertical sides leaves both finger pads above the support plane.
    ids=np.flatnonzero((points[:,1]>.002)&(points[:,0]>.04)&(points[:,2]>.015)&(np.abs(normals[:,1])<.7))
    if len(ids)<10:return None
    if len(ids)>1800:ids=ids[np.linspace(0,len(ids)-1,1800).astype(int)]
    p=points[ids];n=normals[ids];v=p[:,None]-p[None,:];sep=np.linalg.norm(v,axis=-1)
    valid=(n@n.T<-.6)&(sep>.006)&(sep<.035)
    valid&=(np.einsum('ijk,ik->ij',v,n)>.7*sep)&(-np.einsum('ijk,jk->ij',v,n)>.7*sep)
    best=None
    for finger in [1,2,3]:
        cost=np.sum((p-tips[0])**2,axis=-1)[:,None]+np.sum((p-tips[finger])**2,axis=-1)[None,:]
        cost[~valid]=np.inf;i,j=np.unravel_index(np.argmin(cost),cost.shape)
        if np.isfinite(cost[i,j]) and (best is None or cost[i,j]<best[0]):best=(cost[i,j],finger,i,j)
    if best is None:return None
    _,finger,i,j=best
    return {0:p[i]+.0005*n[i],finger:p[j]+.0005*n[j]}, {'opposing_fingers':['thumb',G.FINGERS[finger]],'normal_dot':float(n[i]@n[j]),'contact_separation_m':float(sep[i,j]),'proposal':'vertical sides of existing raised corner marker'}

def generate(spec,out):
    exp=Experiment(spec);m=exp.model;ref=exp.ref;Q=exp.base.copy();N=len(Q);fps=exp.fps
    d=mujoco.MjData(m)
    passive=servo_rollout(m,np.repeat(Q[:1],80,axis=0),fps,qpos_start=exp.initial,record_substeps=False)
    np.savez_compressed(out/'passive_forecast.npz',**passive)
    if np.max(passive['hand_object_force_n'])>1e-6:raise ValueError('Passive forecast touched an object')
    allseeds={};report=[]
    for side,(name,frames,begin,close,end) in SIDE_SPEC.items():
        rig=H.HandRig.build(m,f'robot_{side}_',side)
        cols=np.r_[[rig.wrist_qpos[k] for k in ['x','y','z','rx','ry','rz']],list(rig.finger_qpos.values())].astype(int)
        oc=exp.objects[name];bid=m.body(name).id
        points,normals,tree=G.surface(Path(spec['source'])/'assets'/f'{name}.obj')
        settled=passive['qpos'][-1,oc:oc+7]
        settledR=Rotation.from_quat(settled[3:],scalar_first=True).as_matrix()
        seeds=[]
        for frame in frames:
            d.qpos[:]=Q[frame];mujoco.mj_kinematics(m,d)
            oldp=d.xpos[bid].copy();oldR=d.xmat[bid].reshape(3,3).copy()
            oldw=rig.wrist_pos(d);oldwr=H.euler_to_mat(rig.wrist_euler(d))
            q0=Q[frame].copy();q0[oc:oc+7]=settled
            q0[cols[:3]]=settled[:3]+settledR@oldR.T@(oldw-oldp)
            q0[cols[3:6]]=G.nearest_euler(H.mat_to_euler(settledR@oldR.T@oldwr),q0[cols[3:6]])
            # Keep the inactive hand at home while constructing the single-hand prior.
            other='right' if side=='left' else 'left';q0[exp.cols[other]]=Q[0,exp.cols[other]]
            for othername,adr in exp.objects.items():q0[adr:adr+7]=passive['qpos'][-1,adr:adr+7]
            d.qpos[:]=q0;mujoco.mj_fwdPosition(m,d)
            sites=[rig.fingertip_sites[f] for f in G.FINGERS]
            tips=(d.site_xpos[sites]-settled[:3])@settledR
            if side=='left' and os.environ.get('E25_LEFT_PROPOSAL')=='peg':chosen=peg_targets(tips,points,normals)
            else:
                chosen=G.opposing_targets(tips,points,normals,tree)
                if chosen is None:chosen=board_targets(tips,points,normals)
            if chosen is None:
                report.append({'side':side,'frame':frame,'failure':'no opposing surfaces'});continue
            targets,opposition=chosen
            targets={k:settled[:3]+settledR@v for k,v in targets.items()}
            q,ik=G.bounded_ik(m,d,rig,q0,targets,bid);d.qpos[:]=q;mujoco.mj_forward(m,d)
            localp=settledR.T@(rig.wrist_pos(d)-settled[:3]);localR=settledR.T@H.euler_to_mat(rig.wrist_euler(d))
            pen=geom_pen(m,d,rig,bid)
            row={'side':side,'frame':frame,**ik,**opposition,'penetration_m':pen,'local_wrist':localp.tolist()}
            report.append(row);print(json.dumps(row),flush=True)
            if ik['max_target_residual_m']>.025 or pen['table']>.004 or pen['target']>.006:continue
            # Test a natural pad target and slightly tighter variants in real physics.
            for press in [0.,.08,.16]:
                fingers=q[cols[6:]].copy()
                for j,jname in enumerate(rig.finger_qpos):
                    if any(x in jname for x in ['MCP','PIP','DIP','IP']) and 'AA' not in jname:
                        fingers[j]+=press
                candidate=Q[:,cols].copy();previous=candidate[0,3:6].copy()
                for t in range(begin,N):
                    f=np.clip((t-close)/10.,0.,1.);f=f*f*(3-2*f)
                    goalR=Rotation.from_quat(Q[t,oc+3:oc+7],scalar_first=True)
                    sR=Rotation.from_matrix(settledR)
                    oR=(sR*Rotation.from_rotvec((sR.inv()*goalR).as_rotvec()*f)).as_matrix()
                    op=(1-f)*settled[:3]+f*Q[t,oc:oc+3]
                    wr=G.nearest_euler(H.mat_to_euler(oR@localR),previous);previous=wr
                    target=np.r_[op+oR@localp,wr,fingers]
                    a=np.clip((t-begin)/max(1,close-begin),0.,1.);a=a*a*(3-2*a)
                    candidate[t]=(1-a)*Q[t,cols]+a*target
                    if t<close:candidate[t,2]+=.02*np.sin(np.pi*a)
                if side=='right':
                    # The observed 5--7s excursion is unnecessary for the left-board
                    # grip and risks a collision. Wait at the initial collision-free
                    # hand pose, then follow the HaWoR approach to board2.
                    for t in range(1,begin):
                        a=np.clip((t-(begin-18))/18.,0.,1.);a=a*a*(3-2*a)
                        candidate[t]=(1-a)*Q[0,cols]+a*Q[t,cols]
                candidate=np.clip(candidate,exp.limits[side][:,0],exp.limits[side][:,1]);candidate[0]=Q[0,cols]
                seeds.append({'frame':frame,'press':press,'commands':candidate,'metadata':row})
        allseeds[side]=seeds
    dump(out/'geometry_candidates.json',report)
    return allseeds

_EXP=None
def init(spec):
    global _EXP
    _EXP=Experiment(spec)
def evaluate(task):
    index,path=task;exp=_EXP
    Q=np.load(path)['qpos'];r=servo_rollout(exp.model,Q,exp.fps,qpos_start=exp.initial,record_substeps=False)
    if r['unstable']:return index,{'cost':1e9,'unstable':True}
    p,q=exp.object_poses(r['qpos']);pe=np.linalg.norm(p-exp.tp,axis=-1).mean(0)
    ae=angular_errors(q[:,None],exp.tq[:,None],exp.axes,exp.kinds)[:,0].mean(0)
    ratio=np.maximum(pe/.03,ae/np.deg2rad(30));pen=float(np.max(r['hand_pen_m']))
    return index,{'cost':float(ratio.max()+.25*ratio.mean()+100*max(0,pen-.01)),
        'mean_position_cm':(100*pe).tolist(),'mean_orientation_deg':np.rad2deg(ae).tolist(),
        'mp_sr':float(np.all((pe<.03)&(ae<np.deg2rad(30)))),'max_hand_pen_m':pen}

def main(a):
    out=a.output;out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((E/'input_manifest.json').read_text())
    (out/'search_physics').mkdir(exist_ok=True)
    _,frozen=_freeze(W/'runs/baseline_v1/episode_000025/baseline/execution/scene_physics.xml',out/'search_physics')
    spec={'scene':str(frozen),'reference':str(E/'tracking_reference.npz'),
        'original_reference':str(E/'tracking_reference.npz'),'source':manifest['source'],'geometry':str(E/'metric_geometry.json')}
    dump(out/'spec.json',spec)
    seeds=generate(spec,out);exp=Experiment(spec)
    if not all(seeds.values()):dump(out/'failed.json',{'reason':'No feasible geometric seeds','counts':{k:len(v) for k,v in seeds.items()}});return
    tasks=[];combos=[]
    for i,(left,right) in enumerate(itertools.product(seeds['left'],seeds['right'])):
        Q=exp.base.copy();Q[:,exp.cols['left']]=left['commands'];Q[:,exp.cols['right']]=right['commands'];Q[0]=exp.base[0]
        for oc in exp.objects.values():assert np.array_equal(Q[:,oc:oc+7],exp.base[:,oc:oc+7])
        path=out/'proposals'/f'{i:04d}.npz';path.parent.mkdir(exist_ok=True)
        TR.save_reference(path,exp.model,Q,exp.ref['labels'],exp.fps,exp.ref['segment'],exp.ref['segments'],source='HaWoR-seeded opposing board grasp; physical evaluation required',arms_coupled=False)
        tasks.append((i,str(path)));combos.append({'left':{k:v for k,v in left.items() if k!='commands'},'right':{k:v for k,v in right.items() if k!='commands'}})
    dump(out/'combinations.json',combos);reports=[]
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('spawn'),initializer=init,initargs=(spec,)) as pool:
        for index,r in pool.map(evaluate,tasks):
            row={'index':index,**r};reports.append(row);dump(out/'progress.json',{'evaluated':len(reports),'total':len(tasks),'best':min(reports,key=lambda x:x['cost'])});print(json.dumps(row),flush=True)
    dump(out/'scores.json',reports);best=min(reports,key=lambda x:x['cost']);path=Path(tasks[best['index']][1])
    execution=execute_reference(spec['scene'],path,out/'execution',expected_duration_s=(len(exp.base)-1)/exp.fps)
    metrics=evaluate_files(manifest['source'],execution['trajectory'],E/'input_manifest.json',execution['scene'],E/'metric_geometry.json')
    dump(out/'result.json',{'episode':'episode_000025','reference':str(path),'best':best,'execution':execution,'metrics':metrics,'learned_policy':False,'object_targets_unchanged':True,'initial_state_unchanged':True,'HaWoR_seed':True})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=8);main(p.parse_args())
