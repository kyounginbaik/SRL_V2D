"""Plan on exposed collision surfaces, then screen actual opposing contacts."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse,json,sys
import numpy as np,mujoco,trimesh
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
B=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(B.parents[1]/'third_party/DexAgent'))
from tool_library.common import hand as H
from grasp_geometry_seeds_v4 import bounded_ik
from lid_grasp_seed import contact_penetration
from predict_settled_grasp import smooth
from wrist_reference import continuous_euler_xyz

SCENE=B/'refinement_20261017/lid_seed/episode_000001/candidate_02/execution/scene_physics.xml'
OUT=B/'refinement_20261017/lid_collision_grasps/episode_000001'
WORK={}


def collision_surface(model,data,body):
    """Sample exposed hull faces, excluding faces buried inside another part."""
    op=data.xpos[body].copy();R=data.xmat[body].reshape(3,3).copy();hulls=[];meshes=[]
    for geom in range(model.ngeom):
        if model.geom_bodyid[geom]!=body or not model.geom_contype[geom]:continue
        mid=int(model.geom_dataid[geom]);start=model.mesh_vertadr[mid];count=model.mesh_vertnum[mid]
        vertices=model.mesh_vert[start:start+count].astype(float)
        vertices=(vertices@data.geom_xmat[geom].reshape(3,3).T+data.geom_xpos[geom]-op)@R
        hull=ConvexHull(vertices);hulls.append(hull)
        mesh=trimesh.Trimesh(vertices=vertices,faces=hull.simplices,process=False);mesh.fix_normals(multibody=True);meshes.append(mesh)
    points=[];normals=[];owners=[]
    for i,mesh in enumerate(meshes):
        p,f=trimesh.sample.sample_surface(mesh,1600,seed=43+i);mask=p[:,0]>.018
        points.extend(p[mask]);normals.extend(mesh.face_normals[f[mask]]);owners.extend([i]*int(mask.sum()))
    points=np.asarray(points);normals=np.asarray(normals);owners=np.asarray(owners)
    visible=np.ones(len(points),bool)
    for i,hull in enumerate(hulls):
        inside=np.all(points@hull.equations[:,:3].T+hull.equations[:,3]<-1e-5,axis=1)
        visible&=~(inside&(owners!=i))
    return points[visible],normals[visible],dict(parts=len(hulls),sampled_raised_points=len(points),exposed_raised_points=int(visible.sum()))


def contact_opposition(model,data,rig,object_body):
    patches={'thumb':[],'index':[]}
    for contact in data.contact:
        a,b=int(model.geom_bodyid[contact.geom1]),int(model.geom_bodyid[contact.geom2])
        hand=b if a==object_body else a if b==object_body else -1
        if hand not in rig.body_ids:continue
        name=model.body(hand).name
        finger='thumb' if '_thumb_' in name else 'index' if '_index_' in name else None
        if finger:
            n=contact.frame[:3].copy()*(1 if a==object_body else -1)
            patches[finger].append(dict(normal=n.tolist(),position=contact.pos.tolist(),penetration_m=max(0.,-float(contact.dist))))
    dots=[float(np.asarray(a['normal'])@np.asarray(b['normal'])) for a in patches['thumb'] for b in patches['index']]
    return dict(patches=patches,min_normal_dot=min(dots) if dots else None)


def init():
    model=mujoco.MjModel.from_xml_path(str(SCENE))
    WORK.update(model=model,data=mujoco.MjData(model),rig=H.HandRig.build(model,'robot_right_','right'))


def solve(job):
    model,data,rig=WORK['model'],WORK['data'],WORK['rig'];body=model.body('white_pot_lid').id
    q,ik=bounded_ik(model,data,rig,np.asarray(job['q0']),{int(k):np.asarray(v) for k,v in job['targets'].items()},body)
    data.qpos[:]=q;mujoco.mj_forward(model,data);pen=contact_penetration(model,data,rig,body)
    opposition=contact_opposition(model,data,rig,body)
    geometry=ik['max_target_residual_m']<.008 and pen['target']<.003 and pen['self']<.003 and pen['table']<.002 and pen['other']<.003
    opposed=opposition['min_normal_dot'] is not None and opposition['min_normal_dot']<-.5
    row={k:v for k,v in job.items() if k not in ('q0','targets')}
    row.update(ik=ik,penetration_m=pen,actual_contacts=opposition,geometry_valid=geometry,opposing_collision_contacts=opposed,
        score=ik['max_target_residual_m']+3*max(pen.values())+.03*ik['wrist_translation_change_m'])
    return row,q


def main(workers=4):
    OUT.mkdir(parents=True,exist_ok=True);model=mujoco.MjModel.from_xml_path(str(SCENE));data=mujoco.MjData(model)
    rig=H.HandRig.build(model,'robot_right_','right');body=model.body('white_pot_lid').id
    ref=dict(np.load(B/'refinement_20261017/settled_seed/episode_000001/geometry_v2_settled_reference.npz'))
    forecast=np.load(B/'refinement_20261017/settled_seed/episode_000001/passive_forecast/trajectory.npz')
    frame=34;fps=float(ref['fps']);q0=ref['qpos'][frame].copy();object_cols=[]
    for name in forecast['object_names']:
        adr=int(model.jnt_qposadr[model.joint(str(name)+'_free').id]);object_cols.extend(range(adr,adr+7))
    q0[object_cols]=forecast['qpos'][frame,object_cols];data.qpos[:]=q0;mujoco.mj_forward(model,data)
    points,normals,surface=collision_surface(model,data,body);op=data.xpos[body].copy();R=data.xmat[body].reshape(3,3).copy()
    tips=(data.site_xpos[[rig.fingertip_sites['thumb'],rig.fingertip_sites['index']]]-op)@R
    ai=np.argsort(np.linalg.norm(points-tips[0],axis=1))[:500];bi=np.argsort(np.linalg.norm(points-tips[1],axis=1))[:700]
    a,b=points[ai],points[bi];delta=a[:,None]-b[None];distance=np.linalg.norm(delta,axis=-1)
    dots=normals[ai]@normals[bi].T
    along_a=np.einsum('ijk,ik->ij',delta,normals[ai]);along_b=np.einsum('ijk,jk->ij',-delta,normals[bi])
    valid=(distance>.006)&(distance<.055)&(dots<-.8)&(along_a>.65*distance)&(along_b>.65*distance)
    cost=np.linalg.norm(a-tips[0],axis=1)[:,None]**2+np.linalg.norm(b-tips[1],axis=1)[None]**2;cost[~valid]=np.inf
    pairs=[]
    for flat in np.argsort(cost.ravel()):
        i,j=np.unravel_index(flat,cost.shape)
        if not np.isfinite(cost[i,j]):break
        pair=np.stack([a[i],b[j]])
        if any(np.linalg.norm(pair-p[0])<.006 for p in pairs):continue
        pairs.append((pair,np.stack([normals[ai[i]],normals[bi[j]]]),float(dots[i,j])))
        if len(pairs)==8:break
    jobs=[]
    for index,(pair,normal,dot) in enumerate(pairs):
        for offset in [.0015,0.]:
            initial=q0.copy()
            for name,col in rig.finger_qpos.items():
                if any(f in name for f in ('middle','ring','pinky')) and name.endswith(('_MCP_FE','_PIP','_DIP')):
                    jid=int(model.actuator_trnid[rig.finger_act[name],0]);initial[col]=np.clip(0.,*model.jnt_range[jid])
            targets=(pair+offset*normal)@R.T+op
            jobs.append(dict(pair=index,pair_local=pair.tolist(),normal_dot=dot,site_offset_m=offset,
                q0=initial.tolist(),targets={0:targets[0].tolist(),1:targets[1].tolist()}))
    results=[]
    with ProcessPoolExecutor(max_workers=workers,initializer=init) as pool:
        for row,q in pool.map(solve,jobs):
            results.append((row,q));print(json.dumps(row),flush=True)
    usable=sorted([item for item in results if item[0]['geometry_valid'] and item[0]['opposing_collision_contacts']],key=lambda x:x[0]['score'])
    exports=[]
    for rank,(row,q) in enumerate(usable[:4]):
        path=OUT/f'pose_{rank:02d}.npz';np.savez_compressed(path,qpos=q,object_cols=object_cols)
        exports.append(dict(rank=rank,pose=str(path),geometry=row))
    report=dict(surface=surface,pairs=len(pairs),solves=len(results),actual_opposing_feasible=len(usable),
        all_candidates=[r[0] for r in results],exported=exports,physical_success_claim=False)
    (OUT/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='all_candidates'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4)
    main(parser.parse_args().workers)
