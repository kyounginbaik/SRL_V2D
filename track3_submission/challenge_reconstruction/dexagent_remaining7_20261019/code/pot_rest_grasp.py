#!/usr/bin/env python3
"""Robot-only approach to the pot's predicted physical resting pose."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,sys
from pathlib import Path
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
from prepare_seven import BASE,LEGACY,ROOT,write
sys.path[:0]=[str(LEGACY/'code'),str(ROOT/'third_party/DexAgent')]
from physics_pickup_search_v2 import PickupExperiment
from floating_runtime import servo_rollout,execute_reference
from evaluate_mp_sr import evaluate_files
from wrist_reference import continuous_euler_xyz
from tool_library.common import tracking as TR,hand as H

def run(a):
 source=BASE/'diagnostics/pot_transfer/left_donor_park_right';out=BASE/'diagnostics/pot_transfer'/a.tag
 if out.exists():raise FileExistsError(out)
 out.mkdir(parents=True);spec=json.loads((source/'spec.json').read_text());model=mujoco.MjModel.from_xml_path(spec['scene']);ref=TR.load_reference(spec['reference'],model);Q=ref['qpos'];fps=ref['fps'];oc=int(model.joint('white_pot_free').qposadr[0]);w=np.arange(6)
 parked=Q[:int(round(a.rest_time*fps))+1].copy();cols=model.jnt_qposadr[model.actuator_trnid[:,0]];parked[:,cols]=parked[0,cols]
 roll=servo_rollout(model,parked,fps,record_substeps=False);rest=roll['qpos'][-1];anchor=np.load(source/'anchor_pose.npz')['qpos'].copy()
 oldR=Rotation.from_quat(anchor[oc+3:oc+7],scalar_first=True);lp=oldR.inv().apply(anchor[w[:3]]-anchor[oc:oc+3]);lr=oldR.inv()*Rotation.from_euler('XYZ',anchor[w[3:]])
 newR=Rotation.from_quat(rest[oc+3:oc+7],scalar_first=True)
 anchor[oc:oc+7]=rest[oc:oc+7];anchor[w[:3]]=rest[oc:oc+3]+newR.apply(lp+np.array(a.local_offset));anchor[w[3:]]=continuous_euler_xyz(Rotation.from_matrix((newR*lr).as_matrix()[None]),Q[0,w[3:]])[0]
 if a.opening:
  fk=mujoco.MjData(model);fk.qpos[:]=anchor;mujoco.mj_forward(model,fk);body=model.body('white_pot').id;normals={}
  rig=H.HandRig.build(model,'robot_left_','left')
  for contact in fk.contact:
   b1,b2=[int(model.geom_bodyid[g]) for g in [contact.geom1,contact.geom2]]
   hand=b2 if b1==body else b1 if b2==body else -1
   if hand<0:continue
   finger=next((f for f in ['thumb','index'] if f'left_{f}_' in model.body(hand).name),None)
   if finger:normals.setdefault(finger,[]).append(np.array(contact.frame[:3])*(1 if b1==body else -1))
  targets={}
  for finger,ns in normals.items():
   normal=np.mean(ns,axis=0);normal/=np.linalg.norm(normal);targets[finger]=fk.site_xpos[rig.fingertip_sites[finger]].copy()+a.opening*normal
  if len(targets)!=2:raise ValueError('Static donor pinch lacks both finger patches')
  H.finger_ik(model,fk,rig,targets,iters=180);spec['open_fingers']=fk.qpos[np.array(list(rig.finger_qpos.values()))].tolist()
 if a.roll_deg:
  fk=mujoco.MjData(model);fk.qpos[:]=anchor;mujoco.mj_forward(model,fk);body=model.body('white_pot').id;points=[]
  for contact in fk.contact:
   b1,b2=[int(model.geom_bodyid[g]) for g in [contact.geom1,contact.geom2]]
   hand=b2 if b1==body else b1 if b2==body else -1
   if hand>=0 and any(f'left_{f}_' in model.body(hand).name for f in ['thumb','index']):points.append(np.array(contact.pos))
  if not points:raise ValueError('No static pinch patch to roll around')
  center=np.mean(points,axis=0);axis=center-(rest[oc:oc+3]+newR.apply(model.body_ipos[body]));axis/=np.linalg.norm(axis)
  delta=Rotation.from_rotvec(axis*np.deg2rad(a.roll_deg));anchor[w[:3]]=center+delta.apply(anchor[w[:3]]-center)
  oldw=Rotation.from_euler('XYZ',anchor[w[3:]]);anchor[w[3:]]=continuous_euler_xyz(Rotation.from_matrix((delta*oldw).as_matrix()[None]),Q[0,w[3:]])[0]
  write(out/'grasp_roll.json',dict(angle_deg=a.roll_deg,world_axis=axis.tolist(),world_contact_center=center.tolist(),reason='Rotate the robot grasp around handle to clear tabletop, never rotate object targets'))
 np.savez_compressed(out/'anchor_pose.npz',qpos=anchor)
 write(out/'settling.json',dict(semantics='Predicted rest pose only changes robot approach commands. Scene home and every object target remain immutable.',initial_object_pose=Q[0,oc:oc+7].tolist(),predicted_object_pose=rest[oc:oc+7].tolist(),rest_time_s=a.rest_time,rest_motion_m=float(np.linalg.norm(rest[oc:oc+3]-Q[0,oc:oc+3])),local_wrist_offset_m=a.local_offset))
 spec['anchor_pose']=str(out/'anchor_pose.npz');spec['times']=dict(start=.05,hover=.30,contact=.50,closed=.75,transport=1.15,release=8.8)
 if a.close_shift:
  for k in ['hover','contact','closed','transport']:spec['times'][k]+=a.close_shift
 spec['score_hold_start_s']=.75;spec['score_track_start_s']=1.25
 if a.outward_hover:
  direction=anchor[w[:3]]-rest[oc:oc+3];direction[2]=0.;direction/=np.linalg.norm(direction);spec['hover_offset']=(a.outward_hover*direction+np.array([0.,0.,.06])).tolist()
 if a.hold_to_end:spec['times']['release']=(len(Q)-1)/fps+.1
 write(out/'spec.json',spec);exp=PickupExperiment(spec);x=np.zeros(exp.dim);Qnew,_=exp.commands(x)
 for c in exp.objects.values():assert np.array_equal(Qnew[:,c:c+7],Q[:,c:c+7])
 reference=TR.save_reference(out/'reference.npz',model,Qnew,ref['labels'],fps,ref['segment'],ref['segments'],source='Measured donor grasp; robot approaches predicted free-object rest pose; no object target or initial state modification',floating_runtime=True,no_retiming=True)
 pickup=exp.evaluate(x,return_trace=True);trace=pickup.pop('trace');np.savez_compressed(out/'pickup_trace.npz',trace=trace);write(out/'pickup.json',pickup)
 execution=execute_reference(spec['scene'],reference,out/'execution');metrics=evaluate_files(spec['source'],execution['trajectory'],spec['manifest'],execution['scene'],spec['geometry'])
 write(out/'result.json',dict(episode='episode_000028',reference=str(reference),execution=execution,metrics=metrics,pickup=pickup,spec=spec,unchanged_object_targets=True,unchanged_initial_state=True,learned_policy=False));print(json.dumps(dict(output=str(out),pickup=pickup,mp_sr=metrics['reference_proxy_mp_sr'])),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--rest-time',type=float,default=.5);p.add_argument('--local-offset',type=float,nargs=3,default=[0.,0.,0.]);p.add_argument('--close-shift',type=float,default=0.);p.add_argument('--roll-deg',type=float,default=0.);p.add_argument('--opening',type=float,default=0.);p.add_argument('--outward-hover',type=float,default=0.);p.add_argument('--hold-to-end',action='store_true');run(p.parse_args())
