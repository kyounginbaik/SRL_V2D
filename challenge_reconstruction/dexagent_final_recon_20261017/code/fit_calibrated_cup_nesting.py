"""Use a physically settled relative cup fit as a robot placement subgoal.

All recorded object initial states and immutable reference trajectories remain
unchanged. The planning subgoal alters only the right wrist/finger commands;
the exact two-cup pickup prefix and left-hand controller are protected.
"""
from pathlib import Path
import json,sys,argparse
import mujoco,numpy as np
from scipy.spatial.transform import Rotation
from physics_cup_pickup_search import blend,dump,TR,H
from wrist_reference import continuous_euler_xyz
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files
from audit_object_support import audit
from audit_full_joint_limits import main as joint_audit

B=Path(__file__).resolve().parents[1];R=B/'refinement_20261018/episode_000010'

def transform(p,q):
    T=np.eye(4);T[:3,:3]=Rotation.from_quat(q,scalar_first=True).as_matrix();T[:3,3]=p;return T

def run(start,complete,output):
    source=R/'calibrated_beige_pickup';scene=source/'execution/scene_physics.xml';out=Path(output);out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():raise FileExistsError(out)
    m=mujoco.MjModel.from_xml_path(str(scene));ref=TR.load_reference(source/'reference.npz',m);Q=ref['qpos'].copy();fps=ref['fps'];k=round(start*fps)
    rig=H.HandRig.build(m,'robot_right_','right');wc=np.array([rig.wrist_qpos[n] for n in ['x','y','z','rx','ry','rz']]);fc=np.array(list(rig.finger_qpos.values()))
    bc=int(m.joint('blue_cup_free').qposadr[0]);oc=int(m.joint('beige_cup_free').qposadr[0])
    with np.load(source/'execution/trajectory.npz') as z:actual=z['qpos'];oldsub=z['substep_qpos'].copy()
    with np.load(R/'calibrated_blue_release/execution/trajectory.npz') as z:blue_forecast=z['qpos'][:,bc:bc+7].copy()
    achieved_obj=transform(actual[k,oc:oc+3],actual[k,oc+3:oc+7]);wrist=np.eye(4);wrist[:3,3]=Q[k,wc[:3]];wrist[:3,:3]=Rotation.from_euler('XYZ',Q[k,wc[3:]]).as_matrix()
    object_from_command=np.linalg.inv(achieved_obj)@wrist
    planning=json.loads((B/'refinement_20261018/nesting_plan/episode_000010/report.json').read_text());candidates=[]
    blue_at_complete=transform(blue_forecast[round(complete*fps),:3],blue_forecast[round(complete*fps),3:])
    for record in planning['settles']:
        if not record['settled_by_declared_criterion']:continue
        poses=record['final_pose'];blue=transform(poses['blue_cup']['position_setup_m'],poses['blue_cup']['quaternion_wxyz']);beige=transform(poses['beige_cup']['position_setup_m'],poses['beige_cup']['quaternion_wxyz'])
        relative=np.linalg.inv(blue)@beige;target=blue_at_complete@relative
        angle=Rotation.from_matrix(achieved_obj[:3,:3].T@target[:3,:3]).magnitude()
        candidates.append((angle,record['candidate_index'],relative))
    angle,index,relative=min(candidates,key=lambda x:x[0]);rot=Rotation.from_euler('XYZ',Q[:,wc[3:]]).as_matrix();startR=Rotation.from_matrix(wrist[:3,:3]);t=np.arange(len(Q))/fps
    for i in range(k+1,len(Q)):
        goal=transform(blue_forecast[i,:3],blue_forecast[i,3:])@relative@object_from_command;a=float(blend((t[i]-start)/(complete-start)))
        Q[i,wc[:3]]=(1-a)*wrist[:3,3]+a*goal[:3,3]
        rot[i]=(startR*Rotation.from_rotvec(a*(startR.inv()*Rotation.from_matrix(goal[:3,:3])).as_rotvec())).as_matrix()
        Q[i,fc]=ref['qpos'][k,fc]
    euler=continuous_euler_xyz(Rotation.from_matrix(rot),Q[0,wc[3:]])
    Q[k+1:,wc[3:]]=euler[k+1:]
    other=np.setdiff1d(np.arange(m.nq),np.r_[wc,fc]);assert np.array_equal(Q[:,other],ref['qpos'][:,other]);assert np.array_equal(Q[:k+1],ref['qpos'][:k+1])
    reference=TR.save_reference(out/'reference.npz',m,Q,ref['labels'],fps,ref['segment'],ref['segments'],source='Robot-only nesting subgoal from physically settled relative geometry; immutable episode object targets unchanged',arms_coupled=False,floating_runtime=True)
    execution=execute_reference(scene,reference,out/'execution');inp=R/'calibrated_inputs/episodes/episode_000010';manifest=json.loads((inp/'input_manifest.json').read_text());metrics=evaluate_files(manifest['source'],execution['trajectory'],inp/'input_manifest.json',execution['scene'],inp/'metric_geometry.json')
    with np.load(execution['trajectory']) as z:err=float(abs(z['substep_qpos'][:round(start/.002)]-oldsub[:round(start/.002)]).max())
    assert err==0.
    result=dict(episode='episode_000010',input_dir=str(inp),reference=str(reference),execution=execution,metrics=metrics,protected_prefix_s=start,protected_prefix_max_qpos_error=err,source_targets_unchanged=True,learned_policy=False,
        placement=dict(start_s=start,complete_s=complete,settled_candidate_index=index,rotation_change_rad=angle,blue_from_beige_goal=relative,planning_forecast=str(R/'calibrated_blue_release/execution/trajectory.npz'),note='Right hand remains closed in this diagnostic; subsequent supported opening must be verified.'))
    dump(out/'result.json',result);joint_audit(out);audit(out);print(json.dumps(metrics['declared_frame_reference_proxy'],indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--start',type=float,default=3.5);p.add_argument('--complete',type=float,default=4.5);p.add_argument('--output',type=Path,required=True);run(**vars(p.parse_args()))
