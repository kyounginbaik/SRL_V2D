"""Opposed handle-face grasp seeds for subsequent dynamic physics CEM."""
from pathlib import Path
import argparse,json,sys
import numpy as np
import mujoco,trimesh
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from physics_pickup_search import dump
from wrist_reference import continuous_euler_xyz

B=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(B.parents[1]/'third_party/DexAgent'))
from tool_library.common import hand as H
from grasp_geometry_seeds_v4 import bounded_ik,FINGERS
from lid_grasp_seed import contact_penetration


def handle_surface(m,d,b):
    p=d.xpos[b].copy();R=d.xmat[b].reshape(3,3).copy();hulls=[];points=[];normals=[];owners=[]
    for g in range(m.ngeom):
        if m.geom_bodyid[g]!=b or not m.geom_contype[g]:continue
        mi=int(m.geom_dataid[g]);v=m.mesh_vert[m.mesh_vertadr[mi]:m.mesh_vertadr[mi]+m.mesh_vertnum[mi]].astype(float)
        v=(v@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]-p)@R
        hull=ConvexHull(v);hulls.append(hull)
        mesh=trimesh.Trimesh(v,hull.simplices,process=False);mesh.fix_normals(multibody=True)
        pts,face=trimesh.sample.sample_surface(mesh,2400,seed=100+len(hulls));keep=pts[:,2]>.11
        points.extend(pts[keep]);normals.extend(mesh.face_normals[face[keep]]);owners.extend([len(hulls)-1]*int(keep.sum()))
    points=np.asarray(points);normals=np.asarray(normals);owners=np.asarray(owners);keep=np.ones(len(points),bool)
    for i,h in enumerate(hulls):keep&=~(np.all(points@h.equations[:,:3].T+h.equations[:,3]<-1e-5,axis=1)&(owners!=i))
    return points[keep],normals[keep]


def prepare(task):
    base=B/'refinement_20261018/episode_000001/pot_inputs'
    out=B/'refinement_20261018/episode_000001/pot_opposed_inputs';out.mkdir(parents=True,exist_ok=True)
    spec=json.loads((base/'pot_01.json').read_text());m=mujoco.MjModel.from_xml_path(spec['scene']);d=mujoco.MjData(m)
    rig=H.HandRig.build(m,'robot_left_','left');b=m.body('white_pot').id;oc=int(m.joint('white_pot_free').qposadr[0])
    q0=np.load(spec['anchor_pose'])['qpos'].copy();old_p=q0[oc:oc+3].copy();old_R=Rotation.from_quat(q0[oc+3:oc+7],scalar_first=True)
    wc=np.array([rig.wrist_qpos[k] for k in ('x','y','z','rx','ry','rz')])
    local_p=old_R.inv().apply(q0[wc[:3]]-old_p);local_R=old_R.inv()*Rotation.from_euler('XYZ',q0[wc[3:]])
    forecast=np.load(B/'refinement_20261017/settled_seed/episode_000001/passive_forecast/trajectory.npz')
    for j in range(m.njnt):
        if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE:
            a=int(m.jnt_qposadr[j]);q0[a:a+7]=forecast['qpos'][30,a:a+7]
    R=Rotation.from_quat(q0[oc+3:oc+7],scalar_first=True);p=q0[oc:oc+3]
    q0[wc[:3]]=p+R.apply(local_p);q0[wc[3:]]=continuous_euler_xyz(Rotation.from_matrix([(R*local_R).as_matrix()]),q0[wc[3:]])[0]
    d.qpos[:]=q0;mujoco.mj_forward(m,d);points,normals=handle_surface(m,d,b)
    z=(.145,.165)[task%2];count=(2,4)[task//2]
    targets={};chosen={}
    for finger in range(count):
        sign=1. if finger==0 else -1.;target=np.array([.017,sign*.015,z+max(0,finger-1)*.023])
        ids=np.flatnonzero(normals[:,1]*sign>.75)
        if not len(ids):raise ValueError('no exposed opposing handle face')
        idx=ids[np.argmin(np.linalg.norm(points[ids]-target,axis=1))]
        targets[finger]=p+R.apply(points[idx]+.0015*normals[idx]);chosen[finger]=(points[idx],normals[idx])
    q,fit=bounded_ik(m,d,rig,q0,targets,b);d.qpos[:]=q;mujoco.mj_forward(m,d)
    penetration=contact_penetration(m,d,rig,b)
    anchor=out/f'pot_anchor_{task:02d}.npz';np.savez_compressed(anchor,qpos=q)
    opened_targets={FINGERS[f]:d.site_xpos[rig.fingertip_sites[FINGERS[f]]].copy()+.004*R.apply(normal)
                    for f,(_,normal) in chosen.items()}
    opening_residual=H.finger_ik(m,d,rig,opened_targets,iters=100)
    candidate=dict(spec,anchor_pose=str(anchor),open_fingers=d.qpos[list(rig.finger_qpos.values())].tolist(),
        grasp_generator='exact exposed convex handle faces, opposed thumb/fingers, whole-hand collision-aware IK',
        selected_fingers=[FINGERS[f] for f in chosen],selected_local_contacts={FINGERS[f]:dict(point=v[0].tolist(),normal=v[1].tolist()) for f,v in chosen.items()},
        ik=fit,opening_ik=opening_residual,proposal_penetration_m=penetration,
        physical_initial_state='exact original scene keyframe; forecast used only for robot proposal',
        scale=np.r_[[.008]*3,[.10]*3,[.015]*3,[.10]*3,[.10]*22,[.12]*22,[.012]*3,[.08]*3,[.10,.08,.12]].tolist())
    dump(out/f'pot_{task:02d}.json',candidate)
    print(json.dumps(dict(task=task,fit=fit,penetration=penetration,opening=opening_residual),indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',type=int,choices=range(4),required=True);prepare(p.parse_args().task)
