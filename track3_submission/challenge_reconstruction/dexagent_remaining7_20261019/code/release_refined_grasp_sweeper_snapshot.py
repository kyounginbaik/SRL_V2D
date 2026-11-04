#!/usr/bin/env python3
"""Open a retained grasp and retreat, with actual full-episode verification."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse,json,os,sys
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import numpy as np
import mujoco
from prepare_seven import BASE,LEGACY,ROOT,write
sys.path.insert(0,str(LEGACY/'code'));sys.path.insert(0,str(ROOT/'third_party/DexAgent'))
from tool_library.common import hand as H,tracking as TR
from physics_pickup_search_v2 import blend
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files

def trial(task):
    source,inp,out,side,obj,release,aperture,retreat,up,opening_origin=task
    source,inp,out=Path(source),Path(inp),Path(out);out.mkdir(parents=True,exist_ok=False)
    report=json.loads((source/'result.json').read_text());manifest=json.loads((inp/'input_manifest.json').read_text())
    scene=report['execution']['scene'];m=mujoco.MjModel.from_xml_path(scene);ref=TR.load_reference(report['reference'],m)
    Q=ref['qpos'].copy();original=Q.copy();times=np.arange(len(Q))/float(ref['fps']);k=round(release*float(ref['fps']))
    rig=H.HandRig.build(m,f'robot_{side}_',side)
    wc=np.array([rig.wrist_qpos[j] for j in ['x','y','z','rx','ry','rz']]);fc=np.array(list(rig.finger_qpos.values()));cols=np.r_[wc,fc]
    oc=int(m.joint(obj+'_free').qposadr[0]);d=mujoco.MjData(m);d.qpos[:]=Q[k];mujoco.mj_forward(m,d)
    targets={}
    pinch_center=.5*(d.site_xpos[rig.fingertip_sites['thumb']]+d.site_xpos[rig.fingertip_sites['index']])
    for finger in H.FINGERS:
        pos=d.site_xpos[rig.fingertip_sites[finger]].copy()
        radial=pos-(pinch_center if opening_origin=='pinch' else Q[k,oc:oc+3])
        if opening_origin=='body':radial[2]=0
        radial/=max(np.linalg.norm(radial),1e-8);targets[finger]=pos+aperture*radial
    if opening_origin=='neutral':
        ids=m.actuator_trnid[list(rig.finger_act.values()),0]
        opened=np.clip(np.zeros(len(fc)),m.jnt_range[ids,0],m.jnt_range[ids,1])
    else:
        H.finger_ik(m,d,rig,targets,iters=180);opened=d.qpos[fc].copy()
    # Stop following noisy end-of-reference motion while the fingers open.
    active=times>=release;Q[np.ix_(active,cols)]=Q[k,cols]
    opening=blend((times-release)/.4)
    Q[:,fc]+=(opened-Q[:,fc])*opening[:,None]
    direction=Q[k,wc[:3]]-Q[k,oc:oc+3];direction[2]=0;direction/=max(np.linalg.norm(direction),1e-8)
    Q[:,wc[:3]]+=(retreat*direction+np.array([0,0,up]))*blend((times-release-.4)/.6)[:,None]
    assert np.isfinite(Q).all()
    other=np.setdiff1d(np.arange(m.nq),cols)
    assert np.array_equal(Q[:,other],original[:,other]);assert np.array_equal(Q[times<release],original[times<release])
    robot=m.jnt_qposadr[m.actuator_trnid[:,0]];Q[:,robot]=np.clip(Q[:,robot],m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1])
    path=TR.save_reference(out/'reference.npz',m,Q,ref['labels'],ref['fps'],ref['segment'],ref['segments'],source='Physically refined grasp with opening and clearance retreat; unchanged object reference',floating_runtime=True)
    execution=execute_reference(scene,path,out/'execution')
    metrics=evaluate_files(manifest['source'],execution['trajectory'],inp/'input_manifest.json',execution['scene'],inp/'metric_geometry.json')
    result=dict(episode=inp.name,input_dir=str(inp),reference=str(path),execution=execution,metrics=metrics,
        release_seconds=release,aperture_m=aperture,retreat_m=retreat,up_m=up,opening_origin=opening_origin,object_targets_unchanged=True,
        parent=str(source),learned_policy=False,submission_ready=False)
    write(out/'result.json',result);print(json.dumps(dict(output=str(out),mp_sr=metrics['reference_proxy_mp_sr'],objects=metrics['official_style_reference_proxy']['objects'])),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--episode-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--side',choices=['left','right'],required=True);p.add_argument('--object',required=True);p.add_argument('--start',type=float,required=True);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--opening-origin',choices=['body','pinch','neutral'],default='body')
    args=p.parse_args();tasks=[]
    for i,(delay,aperture,retreat,up) in enumerate([(0,.015,.08,.03),(.5,.022,.10,.03),(0,.025,0.,.15),(.5,.025,.05,.12)]):
        tasks.append((str(args.source),str(args.episode_dir),str(args.output/f'trial{i}'),args.side,args.object,args.start+delay,aperture,retreat,up,args.opening_origin))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:results=list(pool.map(trial,tasks))
    write(args.output/'summary.json',[dict(reference=r['reference'],metrics=r['metrics']) for r in results])
