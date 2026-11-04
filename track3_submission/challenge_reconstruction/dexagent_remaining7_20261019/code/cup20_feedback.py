#!/usr/bin/env python3
"""Use measured cup-axis error to refine robot commands, then test fresh physics.

The input reconstruction requests up to14mm tabletop penetration late in this
episode. Only robot goals gain clearance; object targets and starts stay fixed.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse,json,os,sys
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from scipy.signal import savgol_filter
from prepare_seven import BASE,LEGACY,ROOT,write
sys.path.insert(0,str(LEGACY/'code'));sys.path.insert(0,str(ROOT/'third_party/DexAgent'))
from tool_library.common import hand as H,tracking as TR
from wrist_reference import continuous_euler_xyz
from physics_pickup_search_v2 import blend
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files

def trial(task):
    source,out,gain,release,side,obj,ramp_start,ramp_duration=task;source=Path(source);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    inp=BASE/'physics_collision_v2/episodes/episode_000020';old=json.loads((source/'result.json').read_text())
    manifest=json.loads((inp/'input_manifest.json').read_text());scene=old['execution']['scene'];m=mujoco.MjModel.from_xml_path(scene)
    ref=TR.load_reference(old['reference'],m);Q=ref['qpos'].copy();saved=Q.copy();fps=float(ref['fps']);times=np.arange(len(Q))/fps
    actual=np.load(old['execution']['trajectory'])['qpos'];assert actual.shape==Q.shape
    rig=H.HandRig.build(m,f'robot_{side}_',side);wc=np.array([rig.wrist_qpos[j] for j in ['x','y','z','rx','ry','rz']]);fc=np.array(list(rig.finger_qpos.values()));cols=np.r_[wc,fc]
    a=int(m.joint(obj+'_free').qposadr[0]);R=Rotation.from_quat(Q[:,a+3:a+7],scalar_first=True);S=Rotation.from_quat(actual[:,a+3:a+7],scalar_first=True)
    axis=np.array(json.loads((inp/'metric_geometry.json').read_text())['objects'][obj]['axis_local'])
    if np.linalg.norm(axis)>1e-8:
        axis/=np.linalg.norm(axis)
        vr=R.apply(np.tile(axis,(len(Q),1)));vs=S.apply(np.tile(axis,(len(Q),1)))
        cross=np.cross(vs,vr);angles=np.arctan2(np.linalg.norm(cross,axis=1),np.sum(vs*vr,axis=1))
        rv=cross/np.maximum(np.linalg.norm(cross,axis=1,keepdims=True),1e-9)*np.minimum(angles,np.radians(28))[:,None]
    else:
        rv=(R*S.inv()).as_rotvec();rv*=np.minimum(1.,np.radians(28)/np.maximum(np.linalg.norm(rv,axis=1),1e-9))[:,None]
    # Extrapolate the proven closed grip past its old9s opening before applying
    # the correction; do not learn from the old opening's subsequent drift.
    k=round(8.5*fps);p=R[k].inv().apply(Q[k,wc[:3]]-Q[k,a:a+3]);r=R[k].inv()*Rotation.from_euler('XYZ',Q[k,wc[3:]])
    after=times>8.5;Q[np.ix_(after,wc[:3])]=Q[after,a:a+3]+R[after].apply(np.tile(p,(after.sum(),1)))
    mats=Rotation.from_euler('XYZ',Q[:,wc[3:]]).as_matrix();mats[after]=(R[after]*r).as_matrix();Q[:,wc[3:]]=continuous_euler_xyz(Rotation.from_matrix(mats),Q[0,wc[3:]])
    Q[np.ix_(after,fc)]=Q[k,fc];rv[after]=rv[k]
    delta=Q[:,a:a+3]-actual[:,a:a+3];delta[after]=delta[k]
    delta=savgol_filter(delta,9,2,axis=0);rv=savgol_filter(rv,9,2,axis=0)
    ramp=blend((times-ramp_start)/ramp_duration);rv*=gain*ramp[:,None];delta*=gain*ramp[:,None]
    D=Rotation.from_rotvec(rv);center=Q[:,a:a+3]
    Q[:,wc[:3]]=center+D.apply(Q[:,wc[:3]]-center)+delta
    Q[:,wc[3:]]=continuous_euler_xyz(D*Rotation.from_euler('XYZ',Q[:,wc[3:]]),Q[0,wc[3:]])
    data=mujoco.MjData(m);body=m.body(obj).id;local=[]
    # Compile the exact collision shell into object-local coordinates once.
    data.qpos[:]=saved[0];mujoco.mj_kinematics(m,data)
    for g in range(m.ngeom):
        if m.geom_bodyid[g]!=body or not m.geom_contype[g]:continue
        mid=int(m.geom_dataid[g]);v=m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid]+m.mesh_vertnum[mid]]
        world=v@data.geom_xmat[g].reshape(3,3).T+data.geom_xpos[g]
        local.append(R[0].inv().apply(world-center[0]))
    local=np.vstack(local);bottom=np.array([(rot.apply(local)+pos)[:,2].min() for rot,pos in zip(R,center)])
    clearance=np.minimum(.03,np.maximum(0,.004-bottom));clearance=savgol_filter(clearance,9,2)
    Q[:,wc[2]]+=np.maximum(0,clearance)*ramp
    if release<times[-1]:
        at=round(release*fps);ids=m.actuator_trnid[list(rig.finger_act.values()),0];opened=np.clip(np.zeros(len(fc)),m.jnt_range[ids,0],m.jnt_range[ids,1])
        active=times>=release;Q[np.ix_(active,cols)]=Q[at,cols]
        Q[:,fc]+=(opened-Q[:,fc])*blend((times-release)/.4)[:,None]
        Q[:,wc[2]]+=.15*blend((times-release-.4)/.6)
    Q[0]=saved[0];other=np.setdiff1d(np.arange(m.nq),cols);assert np.array_equal(Q[:,other],saved[:,other]);assert np.isfinite(Q).all()
    robot=m.jnt_qposadr[m.actuator_trnid[:,0]];Q[:,robot]=np.clip(Q[:,robot],m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1])
    path=TR.save_reference(out/'reference.npz',m,Q,ref['labels'],fps,ref['segment'],ref['segments'],source='Robot-only correction from measured cup-axis feedback and table clearance',floating_runtime=True)
    execution=execute_reference(scene,path,out/'execution');metrics=evaluate_files(manifest['source'],execution['trajectory'],inp/'input_manifest.json',execution['scene'],inp/'metric_geometry.json')
    result=dict(episode=inp.name,input_dir=str(inp),reference=str(path),execution=execution,metrics=metrics,
        parent=str(source),gain=gain,side=side,object=obj,ramp_start=ramp_start,ramp_duration=ramp_duration,release_seconds=release,maximum_clearance_m=float(clearance.max()),object_targets_unchanged=True,other_hand_commands_unchanged=True,learned_policy=False,submission_ready=False)
    write(out/'result.json',result);print(json.dumps(dict(output=str(out),mp_sr=metrics['reference_proxy_mp_sr'],objects=metrics['official_style_reference_proxy']['objects'])),flush=True);return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--side',choices=['left','right'],default='left');p.add_argument('--object',default='blue_cup');p.add_argument('--ramp-start',type=float,default=.1);p.add_argument('--ramp-duration',type=float,default=.5);args=p.parse_args()
    tasks=[(str(args.source),str(args.output/f'gain{g}_release{r}'),g,r,args.side,args.object,args.ramp_start,args.ramp_duration) for g in [.5,1.] for r in [9.,11.]]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:results=list(pool.map(trial,tasks))
    write(args.output/'summary.json',[dict(reference=r['reference'],metrics=r['metrics']) for r in results])
