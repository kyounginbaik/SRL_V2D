#!/usr/bin/env python3
"""Test a measured sweeper grasp with unchanged episode-30 object targets."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from prepare_seven import BASE, ROOT, LEGACY, sha, write
sys.path.insert(0, str(LEGACY/'code'))
sys.path.insert(0, str(ROOT/'third_party/DexAgent'))
from tool_library.common import hand as H, tracking as TR
from transfer_observed_grasp import hand_joints
from wrist_reference import continuous_euler_xyz
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files
from physics_pickup_search_v2 import PickupExperiment
from spoon_transfer import rigid_alignment

DONOR=ROOT/'challenge_reconstruction/episode37_ilchold_submission_20261019/common_components/episode_000019'


def run(anchor, park_left, tag):
    work=BASE/'physics_collision_v2'; inp=work/'episodes/episode_000030'
    out=work/'sweeper_transfer'/tag
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    manifest=json.loads((inp/'input_manifest.json').read_text())
    donor_report=json.loads((DONOR/'completed.json').read_text())
    donor_mesh=Path(donor_report['metrics']['geometry']['objects']['mini_sweeper']['mesh_path'])
    target_mesh=Path(manifest['source'])/'assets/mini_sweeper.obj'
    transform, registration=rigid_alignment(donor_mesh,target_mesh,out/'registration.json')
    donor_scene=donor_report['metrics']['provenance']['scene']
    donor_record=donor_report['actions']['path']
    sm=mujoco.MjModel.from_xml_path(donor_scene); sr=np.load(donor_record)
    sj=hand_joints(sm,'right'); sc=sm.jnt_qposadr[sj].astype(int)
    sa=int(sm.joint('mini_sweeper_free').qposadr[0]); sb=sm.body('mini_sweeper').id
    rig=H.HandRig.build(sm,'robot_right_','right')
    k=round(anchor*float(sr['fps'])); d=mujoco.MjData(sm)
    d.qpos[:]=sr['qpos'][k]; d.qvel[:]=sr['qvel'][k]; d.ctrl[:]=sr['ctrl'][k]
    mujoco.mj_forward(sm,d)
    normals={}
    for c in d.contact:
        b1,b2=int(sm.geom_bodyid[c.geom1]),int(sm.geom_bodyid[c.geom2])
        hand=b2 if b1==sb else b1 if b2==sb else -1
        if hand<0:continue
        for finger in H.FINGERS:
            if f'_{finger}_' in sm.body(hand).name:
                normals.setdefault(finger,[]).append(np.array(c.frame[:3])*(1 if b1==sb else -1))
    if 'thumb' not in normals or len(normals)<2:
        raise ValueError('Requested donor frame lacks thumb plus opposing-digit contact: '+str(normals.keys()))
    object_r=Rotation.from_quat(d.qpos[sa+3:sa+7],scalar_first=True)
    local_p=object_r.inv().apply(d.qpos[sc[:3]]-d.qpos[sa:sa+3])
    local_r=object_r.inv()*Rotation.from_euler('XYZ',d.qpos[sc[3:6]])
    aids={int(j):a for a,j in enumerate(sm.actuator_trnid[:,0])}
    closed=np.array([sr['ctrl'][k,aids[j]] for j in sj[6:]])
    targets={}
    for finger, values in normals.items():
        normal=np.mean(values,axis=0); normal/=np.linalg.norm(normal)
        targets[finger]=d.site_xpos[rig.fingertip_sites[finger]].copy()+.008*normal
    H.finger_ik(sm,d,rig,targets,iters=180)
    opened=d.qpos[sc[6:]].copy()
    local_p=transform[:3,:3]@local_p+transform[:3,3]
    local_r=Rotation.from_matrix(transform[:3,:3])*local_r
    baseline=json.loads((work/'runs/baseline_v1/episode_000030/baseline/result.json').read_text())
    scene=baseline['execution']['scene']; model=mujoco.MjModel.from_xml_path(scene)
    tj=hand_joints(model,'right'); cols=model.jnt_qposadr[tj].astype(int)
    if [model.joint(j).name for j in tj]!=[sm.joint(j).name for j in sj]:
        raise ValueError('Donor hand joint order changed')
    original=TR.load_reference(inp/'tracking_reference.npz',model); q=original['qpos'].copy()
    fps=float(original['fps']); times=np.arange(len(q))/fps
    left=model.jnt_qposadr[hand_joints(model,'left')].astype(int)
    # Both reference hands have a spurious post-release excursion. Keep their
    # bounded commands continuous; no object state or timestamp is altered.
    tail=round(11.9*fps); q[tail:,cols]=q[tail,cols]; q[tail:,left]=q[tail,left]
    if park_left:q[:,left]=q[0,left]
    seed=TR.save_reference(out/'seed.npz',model,q,original['labels'],fps,original['segment'],original['segments'],
        source='HaWoR prior with robot-only post-release tail held before tracking outlier',floating_runtime=True,no_retiming=True)
    oc=int(model.joint('mini_sweeper_free').qposadr[0])
    # The approach aims at a measured settled pose, while the transport and
    # all scoring still use the unchanged reference object trajectory.
    measured=np.load(baseline['execution']['trajectory'])['qpos']
    pose=q[0].copy(); settle=round(1.25*fps); pose[oc:oc+7]=measured[settle,oc:oc+7]
    object_r=Rotation.from_quat(pose[oc+3:oc+7],scalar_first=True)
    pose[cols[:3]]=pose[oc:oc+3]+object_r.apply(local_p)
    pose[cols[3:6]]=continuous_euler_xyz(Rotation.from_matrix((object_r*local_r).as_matrix()[None]),q[0,cols[3:6]])[0]
    pose[cols[6:]]=closed
    np.savez_compressed(out/'anchor_pose.npz',qpos=pose)
    lift=np.flatnonzero(original['qpos'][:,oc+2]-original['qpos'][0,oc+2]>.02)[0]/fps
    spec=dict(episode='episode_000030',scene=scene,reference=str(seed),original_reference=str(inp/'tracking_reference.npz'),
        source=manifest['source'],geometry=str(inp/'metric_geometry.json'),manifest=str(inp/'input_manifest.json'),
        object_name='mini_sweeper',side='right',anchor_pose=str(out/'anchor_pose.npz'),open_fingers=opened.tolist(),
        times=dict(start=max(.1,lift-1.5),hover=lift-.9,contact=lift-.6,closed=lift-.25,transport=lift+.15,release=11.9),
        hover_offset=[0.,0.,.06],horizon_s=min(5.,(len(q)-1)/fps),score_hold_start_s=lift-.25,score_track_start_s=lift+.35,
        donor_record=donor_record,donor_record_sha256=sha(donor_record),donor_anchor_s=anchor,donor_contact_digits=list(normals),
        donor_to_target_registration=registration,settled_approach_seconds=1.25,unchanged_original_object_targets=True,
        left_hand_parked=park_left,robot_only_tail_hold_start_s=11.9)
    write(out/'spec.json',spec)
    exp=PickupExperiment(spec); commands,timing=exp.commands(np.zeros(exp.dim))
    all_objects=np.concatenate([np.arange(a,a+7) for a in exp.objects.values()])
    if not np.array_equal(commands[:,all_objects],original['qpos'][:,all_objects]):
        raise AssertionError('Object targets changed')
    reference=TR.save_reference(out/'reference.npz',model,commands,original['labels'],fps,original['segment'],original['segments'],
        source='Measured physically supported episode19 sweeper grasp registered by geometry; robot-only command proposal',floating_runtime=True,no_retiming=True)
    pickup=exp.evaluate(np.zeros(exp.dim),return_trace=True); trace=pickup.pop('trace',None)
    if trace is not None:np.savez_compressed(out/'pickup_trace.npz',trace=trace)
    execution=execute_reference(scene,reference,out/'execution')
    metrics=evaluate_files(manifest['source'],execution['trajectory'],inp/'input_manifest.json',execution['scene'],inp/'metric_geometry.json')
    result=dict(episode='episode_000030',input_dir=str(inp),reference=str(reference),execution=execution,metrics=metrics,
        pickup_diagnostic=pickup,donor=str(DONOR),donor_anchor_s=anchor,left_hand_parked=park_left,
        all_original_object_targets_unchanged=True,original_initial_state_unchanged=True,learned_policy=False,
        submission_ready=False,producer_script_sha256=sha(__file__))
    write(out/'result.json',result)
    print(json.dumps(dict(output=str(out),mp_sr=metrics['reference_proxy_mp_sr'],
        objects=metrics['official_style_reference_proxy']['objects'],pickup=pickup)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--anchor',type=float,required=True);p.add_argument('--park-left',action='store_true');p.add_argument('--tag',required=True)
    run(**vars(p.parse_args()))
