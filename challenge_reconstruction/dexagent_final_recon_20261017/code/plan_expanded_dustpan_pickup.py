#!/usr/bin/env python3
"""Planning-only opposing collision-surface seeds for the dustpan ring.

The static IK poses are proposals, never physical demonstrations or success
results. Every later test still starts from the original full episode state.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse,json,os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
from physics_pickup_search_v2 import B,H,TR,dump
from spoon_collision_grasps import collision_surface,contact_opposition
from grasp_geometry_seeds_v8 import bounded_ik
from lid_grasp_seed import contact_penetration
from wrist_reference import continuous_euler_xyz

OUT=B/'refinement_20261018/expanded_batch'
WORK={}


def initialize(scene,side,object_name):
    m=mujoco.MjModel.from_xml_path(scene)
    WORK.update(m=m,d=mujoco.MjData(m),rig=H.HandRig.build(m,f'robot_{side}_',side),object_name=object_name)


def solve(job):
    m,d,rig=WORK['m'],WORK['d'],WORK['rig'];body=m.body(WORK['object_name']).id
    targets={int(k):np.asarray(v) for k,v in job['targets'].items()}
    q,ik=bounded_ik(m,d,rig,np.asarray(job['q0']),targets,body)
    d.qpos[:]=q;mujoco.mj_forward(m,d);pen=contact_penetration(m,d,rig,body);opp=contact_opposition(m,d,rig,body)
    geometry=ik['max_target_residual_m']<.008 and pen['target']<.003 and pen['self']<.003 and pen['table']<.0005 and pen['other']<.003
    row=dict(pair=job['pair'],offset=job['offset'],ik=ik,penetration_m=pen,opposition=opp,geometry_valid=geometry,
        opposed=opp['min_normal_dot'] is not None and opp['min_normal_dot']<-.5,
        score=ik['max_target_residual_m']+3*max(pen.values())+.03*ik['wrist_translation_change_m'])
    return row,q


def main(workers,episode=38,object_name='mini_dust_pan',side='right'):
    name=f'episode_{episode:06d}';inp=B/'episodes'/name;base=OUT/'native_mpc'/name
    out=OUT/'dustpan_geometry'/name if episode==38 and object_name=='mini_dust_pan' and side=='right' else OUT/'handle_geometry'/name/f'{object_name}_{side}'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'report.json').exists():raise FileExistsError(out/'report.json')
    record=json.loads((base/'result.json').read_text());scene=record['execution']['scene']
    m=mujoco.MjModel.from_xml_path(scene);d=mujoco.MjData(m);rig=H.HandRig.build(m,f'robot_{side}_',side)
    ref=TR.load_reference(record['search']['reference'],m);Q=ref['qpos'];fps=ref['fps'];body=m.body(object_name).id;oc=int(m.joint(object_name+'_free').qposadr[0])
    lift=np.flatnonzero(Q[:,oc+2]-Q[0,oc+2]>.02);start=lift[0]/fps;end=lift[-1]/fps;anchor=round((start-.2)*fps)
    # Measured pre-contact rest geometry comes from the existing causal physical
    # rollout. It is used only for static planning; reference targets stay exact.
    measured=np.load(record['execution']['trajectory']);rest=measured['qpos'][round(.5*fps)]
    q0=Q[anchor].copy();q0[oc:oc+7]=rest[oc:oc+7]
    oldR=Rotation.from_quat(Q[anchor,oc+3:oc+7],scalar_first=True);newR=Rotation.from_quat(rest[oc+3:oc+7],scalar_first=True)
    wc=np.array([rig.wrist_qpos[k] for k in ['x','y','z','rx','ry','rz']]);rot=newR*oldR.inv()
    q0[wc[:3]]=rest[oc:oc+3]+rot.apply(Q[anchor,wc[:3]]-Q[anchor,oc:oc+3])
    q0[wc[3:]]=continuous_euler_xyz(Rotation.from_matrix((rot*Rotation.from_euler('XYZ',Q[anchor,wc[3:]])).as_matrix()[None]),Q[anchor,wc[3:]])[0]
    d.qpos[:]=q0;mujoco.mj_forward(m,d);points,normals,surface=collision_surface(m,d,body)
    op=d.xpos[body].copy();R=d.xmat[body].reshape(3,3).copy();world=points@R.T+op
    keep=world[:,2]>.004;points=points[keep];normals=normals[keep]
    tips=(d.site_xpos[[rig.fingertip_sites['thumb'],rig.fingertip_sites['index']]]-op)@R
    ai=np.argsort(np.linalg.norm(points-tips[0],axis=1))[:1200];bi=np.argsort(np.linalg.norm(points-tips[1],axis=1))[:1600]
    a,b=points[ai],points[bi];delta=a[:,None]-b[None];dist=np.linalg.norm(delta,axis=-1);dots=normals[ai]@normals[bi].T
    valid=(dist>.007)&(dist<.10)&(dots<-.75)
    valid&=np.einsum('ijk,ik->ij',delta,normals[ai])>.65*dist
    valid&=np.einsum('ijk,jk->ij',-delta,normals[bi])>.65*dist
    costs=np.linalg.norm(a-tips[0],axis=1)[:,None]**2+np.linalg.norm(b-tips[1],axis=1)[None]**2;costs[~valid]=np.inf
    pairs=[]
    for flat in np.argsort(costs.ravel()):
        i,j=np.unravel_index(flat,costs.shape)
        if not np.isfinite(costs[i,j]):break
        pair=np.stack([a[i],b[j]])
        if any(np.linalg.norm(pair-old[0])<.006 for old in pairs):continue
        pairs.append((pair,np.stack([normals[ai[i]],normals[bi[j]]])))
        if len(pairs)==12:break
    jobs=[]
    for i,(pair,normal) in enumerate(pairs):
        for offset in [0.,.0015]:
            seed=q0.copy();seed[wc[:3]]+=(pair.mean(0)@R.T+op)-np.mean(d.site_xpos[[rig.fingertip_sites['thumb'],rig.fingertip_sites['index']]],axis=0)
            targets=(pair+offset*normal)@R.T+op
            jobs.append(dict(pair=i,offset=offset,q0=seed.tolist(),targets={0:targets[0].tolist(),1:targets[1].tolist()}))
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize,initargs=(scene,side,object_name)) as pool:
        results=list(pool.map(solve,jobs))
    ranked=sorted(results,key=lambda x:(not x[0]['geometry_valid'],not x[0]['opposed'],x[0]['score']))
    exports=[]
    for rank,(row,q) in enumerate([x for x in ranked if x[0]['geometry_valid']][:6]):
        d.qpos[:]=q;mujoco.mj_forward(m,d);normal=pairs[row['pair']][1]@R.T
        opening={f:d.site_xpos[rig.fingertip_sites[f]].copy()+.006*normal[i] for i,f in enumerate(['thumb','index'])}
        H.finger_ik(m,d,rig,opening,iters=120);opened=d.qpos[list(rig.finger_qpos.values())].copy()
        pose=out/f'pose_{rank:02d}.npz';np.savez_compressed(pose,qpos=q)
        times=dict(start=max(.05,start-1.3),hover=start-.9,contact=start-.5,closed=start-.2,transport=start+.4,release=min((len(Q)-1)/fps-.35,end+.8))
        manifest=json.loads((inp/'input_manifest.json').read_text())
        spec=dict(episode=name,scene=scene,reference=record['search']['reference'],original_reference=str(inp/'tracking_reference.npz'),
            source=manifest['source'],geometry=str(inp/'metric_geometry.json'),manifest=str(inp/'input_manifest.json'),
            object_name=object_name,side=side,anchor_pose=str(pose),open_fingers=opened.tolist(),times=times,
            horizon_s=start+2.2,hover_offset=[0,0,.07],score_hold_start_s=times['closed'],score_track_start_s=times['transport'],
            object_targets_changed=False,planning_only_static_seed=True)
        path=out/f'spec_{rank:02d}.json';dump(path,spec);exports.append(dict(spec=str(path),geometry=row))
    report=dict(episode=name,surface=surface,pairs=len(pairs),solves=len(results),all_candidates=[x[0] for x in ranked],exported=exports,
        physical_success_claim=False,source_object_targets_changed=False)
    dump(out/'report.json',report);print(json.dumps(dict(output=str(out),exports=exports),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=8);p.add_argument('--episode',type=int,default=38)
    p.add_argument('--object-name',default='mini_dust_pan');p.add_argument('--side',choices=['left','right'],default='right');main(**vars(p.parse_args()))
