#!/usr/bin/env python3
"""Propose a physically measured cup grasp in the new HaWoR episode14 scene.

This edits robot commands only. Geometry registration transfers a measured
opposing-contact grasp; fresh dynamics determine whether that seed works.
"""
from pathlib import Path
import argparse
import json
import os
import sys
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '1'
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from spoon_transfer import rigid_alignment
from prepare_seven import BASE, LEGACY, ROOT, write
sys.path.insert(0, str(LEGACY / 'code'))
sys.path.insert(0, str(ROOT / 'third_party/DexAgent'))
from prepare_calibrated_cup_pickups import donor
from tool_library.common import hand as H, tracking as TR
from wrist_reference import continuous_euler_xyz
from physics_cup_pickup_search import PickupExperiment
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files


def run(args):
    inp = args.episode_dir.resolve()
    if inp.name != 'episode_000014':
        raise ValueError('This initial pickup proposal is scoped to episode14')
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    manifest = json.loads((inp / 'input_manifest.json').read_text())
    if args.scene:
        scene = str(args.scene.resolve())
    else:
        baseline = json.loads((args.baseline / 'result.json').read_text())
        scene = baseline['execution']['scene']
    m = mujoco.MjModel.from_xml_path(scene)
    ref = TR.load_reference(inp / 'tracking_reference.npz', m)
    q = ref['qpos']; fps = float(ref['fps']); duration = (len(q)-1)/fps
    source = LEGACY / 'refinement_20261017/cup_observed_retention/episode_000010/a1.9_clear0.015'
    measured = donor(source, 'left', 'blue_cup', 2.5)
    if not measured['qualified_opposing_contact']:
        raise ValueError('Donor does not have a measured opposing grasp')
    old_manifest = json.loads((LEGACY / 'episodes/episode_000010/input_manifest.json').read_text())
    transform, registration = rigid_alignment(
        Path(old_manifest['source']) / 'assets/blue_cup.obj',
        Path(manifest['source']) / 'assets/blue_cup.obj', out / 'mesh_registration.json')
    localp = transform[:3,:3] @ np.array(measured['object_from_measured_wrist_p']) + transform[:3,3]
    localr = Rotation.from_matrix(transform[:3,:3]) * Rotation.from_quat(
        measured['object_from_measured_wrist_q'], scalar_first=True)
    rig = H.HandRig.build(m, 'robot_left_', 'left')
    wc = np.array([rig.wrist_qpos[k] for k in ['x','y','z','rx','ry','rz']])
    fc = np.array(list(rig.finger_qpos.values()))
    limits = m.jnt_range[m.actuator_trnid[list(rig.finger_act.values()),0]]
    oc = int(m.joint('blue_cup_free').qposadr[0])
    rest = q[0].copy()
    if args.rest_time is not None:
        recording = np.load(Path(scene).parent/'trajectory.npz',allow_pickle=False)
        rest = recording['qpos'][round(args.rest_time*float(recording['fps']))].copy()
    r0 = Rotation.from_quat(rest[oc+3:oc+7], scalar_first=True)
    axis = np.array(json.loads((inp/'metric_geometry.json').read_text())['objects']['blue_cup']['axis_local'])
    axis /= np.linalg.norm(axis)
    lift = np.flatnonzero(q[:,oc+2] - q[0,oc+2] > .02)
    if not len(lift):
        raise ValueError('No source cup lift')
    lift_s = float(lift[0]/fps)
    target_local = r0.inv().apply(q[max(0,int((lift_s-.3)*fps)),wc[:3]] - rest[oc:oc+3])
    a = localp-axis*(axis@localp); a/=np.linalg.norm(a)
    b = target_local-axis*(axis@target_local); b/=np.linalg.norm(b)
    angle = np.arctan2(axis@np.cross(a,b),a@b) + np.radians(args.angle_deg)
    symmetry = Rotation.from_rotvec(axis*angle)
    localp = symmetry.apply(localp); localr = symmetry*localr
    anchor = q[0].copy()
    anchor[oc:oc+7] = rest[oc:oc+7]
    anchor[wc[:3]] = rest[oc:oc+3] + r0.apply(localp)
    anchor[wc[3:]] = continuous_euler_xyz(Rotation.from_matrix((r0*localr).as_matrix()[None]),q[0,wc[3:]])[0]
    fingers = np.array(measured['measured_finger_q'])
    pressure = np.clip(np.array(measured['commanded_finger_q'])-fingers,-.04,.04)
    anchor[fc] = np.clip(fingers+pressure,limits[:,0],limits[:,1])
    d = mujoco.MjData(m); d.qpos[:]=anchor; mujoco.mj_forward(m,d)
    targets = {}
    for finger, force in measured['normal_forces_n'].items():
        if force <= .1:
            continue
        p = d.site_xpos[rig.fingertip_sites[finger]].copy()
        radial = r0.inv().apply(p-rest[oc:oc+3]); radial -= axis*(axis@radial)
        radial /= max(np.linalg.norm(radial),1e-8)
        targets[finger] = p + args.open_gap*r0.apply(radial)
    H.finger_ik(m,d,rig,targets,iters=120)
    opened=d.qpos[fc].copy()
    np.savez_compressed(out/'anchor_pose.npz',qpos=anchor)
    contact=max(.4,lift_s-.45)+args.close_shift
    times=dict(start=.05,hover=contact-.2,contact=contact,closed=contact+.22,
               transport=max(contact+.5,lift_s+.2),
               release=duration+1 if args.hold_through_end else min(args.release,duration-.4))
    proposal=q.copy()
    if args.park_right:
        from transfer_observed_grasp import hand_joints
        other=m.jnt_qposadr[hand_joints(m,'right')].astype(int)
        proposal[:,other]=proposal[0,other]
    base=TR.save_reference(out/'base.npz',m,proposal,ref['labels'],fps,ref['segment'],ref['segments'],
        source='HaWoR prior; optional other-hand clearance proposal',floating_runtime=True)
    approach = r0.apply(localp-axis*(axis@localp))
    approach[2] = 0.; approach /= max(np.linalg.norm(approach),1e-8)
    hover_offset = args.hover_radial*approach + np.array([0.,0.,args.hover_up])
    spec=dict(episode=inp.name,scene=scene,reference=str(base),
        original_reference=str(inp/'tracking_reference.npz'),source=manifest['source'],
        geometry=str(inp/'metric_geometry.json'),manifest=str(inp/'input_manifest.json'),
        object_name='blue_cup',side='left',anchor_pose=str(out/'anchor_pose.npz'),
        open_fingers=opened.tolist(),times=times,hover_offset=hover_offset.tolist(),
        measured_rest_time_s=args.rest_time,park_right=args.park_right,
        horizon_s=min(duration,lift_s+2),score_hold_start_s=times['closed'],
        score_track_start_s=lift_s+.3,donor=measured,grasp_rotation_rad=float(angle))
    write(out/'spec.json',spec)
    trial=PickupExperiment(spec); commands,timing=trial.commands(np.zeros(trial.dim))
    objcols=np.concatenate([np.arange(a,a+7) for a in trial.objects.values()])
    if not np.array_equal(commands[:,objcols],q[:,objcols]):
        raise AssertionError('Object targets changed')
    path=TR.save_reference(out/'reference.npz',m,commands,ref['labels'],fps,ref['segment'],ref['segments'],
        source='Measured cup-grasp command seed; HaWoR prior outside left pickup; unchanged objects and clock',floating_runtime=True)
    execution=execute_reference(scene,path,out/'execution')
    metrics=evaluate_files(manifest['source'],execution['trajectory'],inp/'input_manifest.json',execution['scene'],inp/'metric_geometry.json')
    result=dict(episode=inp.name,input_dir=str(inp),reference=str(path),execution=execution,
        metrics=metrics,donor=measured,registration=registration['best'],timing=timing,
        original_object_targets_unchanged=True,learned_policy=False,submission_ready=False)
    write(out/'result.json',result)
    print(json.dumps(dict(output=str(out),mp_sr=metrics['reference_proxy_mp_sr'],objects=metrics['official_style_reference_proxy']['objects'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episode-dir',type=Path,required=True)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--baseline',type=Path)
    source.add_argument('--scene',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--angle-deg',type=float,default=0.)
    p.add_argument('--close-shift',type=float,default=0.)
    p.add_argument('--release',type=float,default=5.)
    p.add_argument('--rest-time',type=float)
    p.add_argument('--park-right',action='store_true')
    p.add_argument('--open-gap',type=float,default=.006)
    p.add_argument('--hover-radial',type=float,default=0.)
    p.add_argument('--hover-up',type=float,default=.06)
    p.add_argument('--hold-through-end',action='store_true')
    run(p.parse_args())
