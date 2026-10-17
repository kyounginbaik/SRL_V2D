"""Handle-specific episode1 grasp proposals at the physically settled lid pose.

Imports the shared v4 collision-aware IK without editing its implementation.
Object states are used only inside geometric planning; exported references and
physical execution retain the original object path and exact initial state.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse,json,sys
import numpy as np,mujoco,trimesh
from scipy.spatial.transform import Rotation
B=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(B.parents[1]/'third_party/DexAgent'))
from tool_library.common import hand as H
from grasp_geometry_seeds_v4 import bounded_ik
from wrist_reference import continuous_euler_xyz
from predict_settled_grasp import smooth
from floating_runtime import execute_reference
from calibrate_controller import controller_scene
from evaluate_mp_sr import evaluate_files

SCENE=B/'runs/pilot_v2/episode_000001/seed_44/baseline/scene_physics.xml'
OUT=B/'refinement_20261017/lid_seed/episode_000001'
WORK={}


def init_worker():
    m=mujoco.MjModel.from_xml_path(str(SCENE))
    WORK.update(model=m,data=mujoco.MjData(m),rig=H.HandRig.build(m,'robot_right_','right'))


def contact_penetration(m,d,rig,target):
    pen={'self':0.,'target':0.,'table':0.,'other':0.}
    for c in d.contact:
        a,b=int(m.geom_bodyid[c.geom1]),int(m.geom_bodyid[c.geom2])
        if a in rig.body_ids and b in rig.body_ids:kind='self'
        elif (a in rig.body_ids)!=(b in rig.body_ids):
            other=b if a in rig.body_ids else a
            kind='target' if other==target else 'table' if m.body(other).name=='table' else 'other'
        else:continue
        pen[kind]=max(pen[kind],-float(c.dist))
    return pen


def solve(job):
    m,d,rig=WORK['model'],WORK['data'],WORK['rig'];target=m.body('white_pot_lid').id
    q0=np.array(job['q0']);targets={int(k):np.array(v) for k,v in job['targets'].items()}
    q,ik=bounded_ik(m,d,rig,q0,targets,target)
    d.qpos[:]=q;mujoco.mj_forward(m,d);pen=contact_penetration(m,d,rig,target)
    good=(ik['max_target_residual_m']<.008 and pen['target']<.003 and pen['self']<.003
          and pen['table']<.002 and pen['other']<.003)
    score=ik['max_target_residual_m']+3*max(pen.values())+.03*ik['wrist_translation_change_m']
    row={k:v for k,v in job.items() if k not in ('q0','targets')}
    row.update(ik=ik,penetration_m=pen,geometry_valid=good,score=float(score))
    return row,q


def handle_pairs(mesh,tips):
    points,faces=trimesh.sample.sample_surface(mesh,60000,seed=43)
    normals=mesh.face_normals[faces]
    raised=points[:,0]>.018;points=points[raised];normals=normals[raised]
    if len(points)<20:raise ValueError('No raised lid handle surface in source mesh')
    ia=np.argsort(np.linalg.norm(points-tips[0],axis=1))[:200]
    ib=np.argsort(np.linalg.norm(points-tips[1],axis=1))[:350]
    a,b=points[ia],points[ib];delta=a[:,None]-b[None]
    distance=np.linalg.norm(delta,axis=-1);normaldot=normals[ia]@normals[ib].T
    outward=np.einsum('ijk,ik->ij',delta,normals[ia])
    valid=(distance>.006)&(distance<.08)&(normaldot<-.3)&(outward>.2*distance)
    cost=np.linalg.norm(a-tips[0],axis=1)[:,None]**2+np.linalg.norm(b-tips[1],axis=1)[None]**2
    cost[~valid]=np.inf;pairs=[]
    for flat in np.argsort(cost.ravel()):
        i,j=np.unravel_index(flat,cost.shape)
        if not np.isfinite(cost[i,j]):break
        pair=np.stack([a[i],b[j]])
        if any(np.linalg.norm(pair-p['points'])<.006 for p in pairs):continue
        pairs.append(dict(points=pair,normals=np.stack([normals[ia[i]],normals[ib[j]]]),cost=float(cost[i,j]),normal_dot=float(normaldot[i,j])))
        if len(pairs)==3:break
    return pairs


def main(workers=4,execute=False):
    OUT.mkdir(parents=True,exist_ok=True);m=mujoco.MjModel.from_xml_path(str(SCENE));d=mujoco.MjData(m)
    rig=H.HandRig.build(m,'robot_right_','right');lid=m.body('white_pot_lid').id
    settled=B/'refinement_20261017/settled_seed/episode_000001'
    base=dict(np.load(settled/'geometry_v2_settled_reference.npz'))
    forecast=dict(np.load(settled/'passive_forecast/trajectory.npz'))
    drift=json.loads((settled/'settled_seed_report.json').read_text())
    corr=next(c for c in drift['corrections'] if c['side']=='right')
    fps=float(base['fps']);N=len(base['qpos']);times=np.arange(N)/fps
    source=B.parent/'final_recon/episode_000001/assets/white_pot_lid_visual.obj'
    mesh=trimesh.load(source,force='mesh',process=False)
    # Correct winding only in this in-memory normal estimate; no asset changes.
    mesh.fix_normals(multibody=True)
    object_cols=[]
    for name in forecast['object_names']:
        adr=int(m.jnt_qposadr[m.joint(str(name)+'_free').id]);object_cols.extend(range(adr,adr+7))
    jobs=[]
    for frame in [34,38]:
        q0=base['qpos'][frame].copy();q0[object_cols]=forecast['qpos'][frame,object_cols]
        d.qpos[:]=q0;mujoco.mj_kinematics(m,d);op=d.xpos[lid].copy();orr=d.xmat[lid].reshape(3,3).copy()
        tips=(d.site_xpos[[rig.fingertip_sites['thumb'],rig.fingertip_sites['index']]]-op)@orr
        pairs=handle_pairs(mesh,tips)
        for pi,pair in enumerate(pairs):
            local=pair['points']+.002*pair['normals'];world=local@orr.T+op
            for opening in ['source','open_other_fingers']:
                initial=q0.copy()
                if opening!='source':
                    for name,col in rig.finger_qpos.items():
                        if any(f in name for f in ('middle','ring','pinky')) and name.endswith(('_MCP_FE','_PIP','_DIP')):
                            jid=int(m.actuator_trnid[rig.finger_act[name],0]);initial[col]=np.clip(0.,*m.jnt_range[jid])
                jobs.append(dict(frame=frame,seconds=frame/fps,pair=pi,opening=opening,
                    pair_local=pair['points'].tolist(),normal_dot=pair['normal_dot'],
                    q0=initial.tolist(),targets={0:world[0].tolist(),1:world[1].tolist()}))
    if not jobs:raise ValueError('No valid opposing handle targets')
    results=[]
    with ProcessPoolExecutor(max_workers=workers,initializer=init_worker) as pool:
        for row,q in pool.map(solve,jobs):
            print(json.dumps(row),flush=True);results.append((row,q))
    feasible=sorted([r for r in results if r[0]['geometry_valid']],key=lambda x:x[0]['score'])
    candidates=[];cols=np.array([rig.wrist_qpos[k] for k in ('x','y','z','rx','ry','rz')]+list(rig.finger_qpos.values()))
    for rank,(row,qbest) in enumerate(feasible[:3]):
        frame=row['frame'];d.qpos[:]=qbest;mujoco.mj_kinematics(m,d)
        op=d.xpos[lid].copy();orr=d.xmat[lid].reshape(3,3).copy()
        localp=orr.T@(qbest[cols[:3]]-op)
        localr=Rotation.from_matrix(orr).inv()*Rotation.from_euler('XYZ',qbest[cols[3:6]])
        q=base['qpos'].copy();original=q.copy()
        begin=.95;full=1.7;end=4.7
        blend=smooth((times-begin)/(full-begin))*(1-smooth((times-(end-.25))/.25))
        drweight=smooth((times-(1.7-.6))/.6)*(1-smooth((times-(1.7+.15))/1.))
        delta=Rotation.from_rotvec(drweight[:,None]*np.array(corr['correction_rotation_vector']))
        desiredp=[];desiredr=[]
        for t in range(N):
            d.qpos[:]=original[t];mujoco.mj_kinematics(m,d)
            objr=Rotation.from_matrix(d.xmat[lid].reshape(3,3))
            p=delta[t].apply(d.xpos[lid]+objr.apply(localp))+drweight[t]*np.array(corr['correction_translation'])
            r=delta[t]*objr*localr
            desiredp.append(p);desiredr.append(r.as_quat())
        desiredp=np.asarray(desiredp);desiredr=Rotation.from_quat(desiredr)
        currentr=Rotation.from_euler('XYZ',original[:,cols[3:6]])
        blended=currentr*Rotation.from_rotvec(blend[:,None]*(currentr.inv()*desiredr).as_rotvec())
        q[:,cols[:3]]=(1-blend[:,None])*original[:,cols[:3]]+blend[:,None]*desiredp
        q[:,cols[3:6]]=continuous_euler_xyz(blended,original[0,cols[3:6]])
        q[:,cols[6:]]=(1-blend[:,None])*original[:,cols[6:]]+blend[:,None]*qbest[cols[6:]]
        q[np.ix_(blend==0,cols)]=original[np.ix_(blend==0,cols)]
        assert np.array_equal(q[:,object_cols],base['qpos'][:,object_cols])
        assert np.array_equal(q[0],base['qpos'][0])
        for j in range(m.njnt):
            if m.joint(j).name.startswith('robot_'):
                adr=int(m.jnt_qposadr[j]);assert np.all(q[:,adr]>=m.jnt_range[j,0]-1e-8) and np.all(q[:,adr]<=m.jnt_range[j,1]+1e-8)
        seed=dict(base,qpos=q);path=OUT/f'candidate_{rank:02d}.npz';np.savez_compressed(path,**seed)
        candidate=dict(rank=rank,reference=str(path),geometry=row,object_reference_unchanged=True,initial_state_unchanged=True,timing_unchanged=True)
        if execute:
            out=OUT/f'candidate_{rank:02d}';out.mkdir(exist_ok=True)
            scene=out/'controller_scene.xml';controller_scene(SCENE,scene,'kp5000_r200_d5')
            run=execute_reference(scene,path,out/'execution')
            metric=evaluate_files(B.parent/'final_recon/episode_000001',run['trajectory'],B/'episodes/episode_000001/input_manifest.json',run['scene'])
            candidate.update(execution=run,metrics=metric)
            print('physics',rank,metric['reference_proxy_mp_sr'],flush=True)
        candidates.append(candidate)
    report=dict(planning='raised-handle thumb/index IK with whole-hand collision residuals at passive-settled object pose',
        candidates_evaluated=len(results),geometrically_feasible=len(feasible),all_candidates=[r[0] for r in results],exported=candidates,
        physical_success_claim=False)
    (OUT/'report.json').write_text(json.dumps(report,indent=2));return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);p.add_argument('--execute',action='store_true');a=p.parse_args();main(a.workers,a.execute)
