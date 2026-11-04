#!/usr/bin/env python3
"""Test a measured pot-handle grip as a robot-only episode28 proposal."""
from pathlib import Path
import argparse,json,os,sys
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from spoon_transfer import rigid_alignment
from prepare_seven import BASE,LEGACY,ROOT,write
sys.path.insert(0,str(LEGACY/'code'));sys.path.insert(0,str(ROOT/'third_party/DexAgent'))
from tool_library.common import hand as H,tracking as TR
from transfer_observed_grasp import hand_joints
from wrist_reference import continuous_euler_xyz
from physics_pickup_search_v2 import PickupExperiment
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files

def run(args):
    inp=args.episode_dir.resolve();out=args.output.resolve()
    if inp.name!='episode_000028':raise ValueError('Episode28 only')
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    manifest=json.loads((inp/'input_manifest.json').read_text())
    source=LEGACY/'refinement_20261018/episode_000003/retention_after45'
    sm=mujoco.MjModel.from_xml_path(str(source/'execution/scene_physics.xml'))
    z=np.load(source/'execution/trajectory.npz',allow_pickle=False)
    k=round(args.anchor*float(z['fps']));d=mujoco.MjData(sm);d.qpos[:]=z['qpos'][k];mujoco.mj_forward(sm,d)
    sj=hand_joints(sm,'left');sc=sm.jnt_qposadr[sj].astype(int);oa=int(sm.joint('white_pot_free').qposadr[0])
    r=Rotation.from_quat(d.qpos[oa+3:oa+7],scalar_first=True)
    lp=r.inv().apply(d.qpos[sc[:3]]-d.qpos[oa:oa+3]);lr=r.inv()*Rotation.from_euler('XYZ',d.qpos[sc[3:6]])
    aid={int(j):i for i,j in enumerate(sm.actuator_trnid[:,0])}
    closed=np.array([z['ctrl'][k,aid[j]] for j in sj[6:]])
    rig=H.HandRig.build(sm,'robot_left_','left');ob=sm.body('white_pot').id;normals={}
    for c in d.contact:
        b1,b2=[int(sm.geom_bodyid[g]) for g in (c.geom1,c.geom2)]
        other=b2 if b1==ob else b1 if b2==ob else -1
        if other not in rig.body_ids:continue
        finger=next((f for f in H.FINGERS if f'_{f}_' in sm.body(other).name),None)
        if finger:normals.setdefault(finger,[]).append(np.array(c.frame[:3])*(1 if b1==ob else -1))
    if 'thumb' not in normals or len(normals)<2:raise ValueError('Donor lacks thumb and another finger contact')
    targets={}
    for f,ns in normals.items():
        normal=np.mean(ns,axis=0);normal/=max(np.linalg.norm(normal),1e-8)
        targets[f]=d.site_xpos[rig.fingertip_sites[f]].copy()+.006*normal
    H.finger_ik(sm,d,rig,targets,iters=120);opened=d.qpos[sc[6:]].copy()
    donor_manifest=json.loads((LEGACY/'episodes/episode_000003/input_manifest.json').read_text())
    transform,registration=rigid_alignment(Path(donor_manifest['source'])/'assets/white_pot.obj',
        Path(manifest['source'])/'assets/white_pot.obj',out/'mesh_registration.json')
    lp=transform[:3,:3]@lp+transform[:3,3];lr=Rotation.from_matrix(transform[:3,:3])*lr
    m=mujoco.MjModel.from_xml_path(str(args.scene.resolve()));ref=TR.load_reference(inp/'tracking_reference.npz',m)
    q=ref['qpos'];fps=float(ref['fps']);duration=(len(q)-1)/fps
    joints=hand_joints(m,'left');cols=m.jnt_qposadr[joints].astype(int)
    if [sm.joint(j).name for j in sj]!=[m.joint(j).name for j in joints]:raise ValueError('Jointorder mismatch')
    a=int(m.joint('white_pot_free').qposadr[0]);r0=Rotation.from_quat(q[0,a+3:a+7],scalar_first=True)
    anchor=q[0].copy();anchor[cols[:3]]=q[0,a:a+3]+r0.apply(lp)
    anchor[cols[3:6]]=continuous_euler_xyz(Rotation.from_matrix((r0*lr).as_matrix()[None]),q[0,cols[3:6]])[0]
    anchor[cols[6:]]=closed;np.savez_compressed(out/'anchor_pose.npz',qpos=anchor)
    proposal=q.copy()
    if args.park_right:
        other=m.jnt_qposadr[hand_joints(m,'right')].astype(int);proposal[:,other]=proposal[0,other]
    base=TR.save_reference(out/'base.npz',m,proposal,ref['labels'],fps,ref['segment'],ref['segments'],source='HaWoR prior; other-hand clearance proposal',floating_runtime=True)
    lift=np.flatnonzero(q[:,a+2]-q[0,a+2]>.02)[0]/fps
    contact=max(.4,lift-.5)+args.close_shift
    times=dict(start=.05,hover=contact-.2,contact=contact,closed=contact+.25,transport=max(lift+.2,contact+.5),release=min(args.release,duration-.4))
    spec=dict(episode=inp.name,scene=str(args.scene.resolve()),reference=str(base),original_reference=str(inp/'tracking_reference.npz'),
        source=manifest['source'],geometry=str(inp/'metric_geometry.json'),manifest=str(inp/'input_manifest.json'),
        object_name='white_pot',side='left',anchor_pose=str(out/'anchor_pose.npz'),open_fingers=opened.tolist(),
        times=times,hover_offset=[0,0,.06],horizon_s=min(duration,lift+2),score_hold_start_s=times['closed'],score_track_start_s=lift+.3)
    write(out/'spec.json',spec);experiment=PickupExperiment(spec);Q,timing=experiment.commands(np.zeros(experiment.dim))
    object_cols=np.concatenate([np.arange(a,a+7) for a in experiment.objects.values()])
    assert np.array_equal(Q[:,object_cols],q[:,object_cols])
    path=TR.save_reference(out/'reference.npz',m,Q,ref['labels'],fps,ref['segment'],ref['segments'],source='Measured pot handle grasp proposal; unchanged source objects and clock',floating_runtime=True)
    execution=execute_reference(args.scene.resolve(),path,out/'execution')
    metrics=evaluate_files(manifest['source'],execution['trajectory'],inp/'input_manifest.json',execution['scene'],inp/'metric_geometry.json')
    result=dict(episode=inp.name,input_dir=str(inp),reference=str(path),execution=execution,metrics=metrics,
        donor=str(source),donor_anchor_s=args.anchor,registration=registration['best'],park_right=args.park_right,
        timing=timing,original_object_targets_unchanged=True,learned_policy=False,submission_ready=False)
    write(out/'result.json',result);print(json.dumps(dict(output=str(out),mp_sr=metrics['reference_proxy_mp_sr'],objects=metrics['official_style_reference_proxy']['objects'])),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episode-dir',type=Path,required=True);p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--anchor',type=float,default=3.5)
    p.add_argument('--park-right',action='store_true');p.add_argument('--close-shift',type=float,default=0.)
    p.add_argument('--release',type=float,default=8.8)
    run(p.parse_args())
