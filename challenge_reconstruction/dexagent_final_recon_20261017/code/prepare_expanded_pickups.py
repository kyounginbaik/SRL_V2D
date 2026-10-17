#!/usr/bin/env python3
"""Transfer a measured held spoon pose into explicit approach/closure/lift searches.

Measured palm geometry and commanded finger preload are kept separate. Contact
normals define a geometric opening, avoiding guessed joint-angle sign changes.
Only robot commands may change; target objects and original time grid stay fixed.
"""
from pathlib import Path
import argparse
import json
import hashlib
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from physics_pickup_search_v2 import B,TR,H,dump
from transfer_observed_grasp import hand_joints
from wrist_reference import continuous_euler_xyz

OUT=B/'refinement_20261018/expanded_batch'
DONOR=B/'refinement_20261018/episode_000003/retention_after45'


def prepare(episodes):
    sm=mujoco.MjModel.from_xml_path(str(DONOR/'execution/scene_physics.xml'))
    sr=H.HandRig.build(sm,'robot_right_','right');measured=np.load(DONOR/'execution/trajectory.npz')
    sc=sm.jnt_qposadr[hand_joints(sm,'right')].astype(int);sa=int(sm.joint('wooden_spoon_free').qposadr[0])
    donor_input=json.loads((B/'episodes/episode_000003/input_manifest.json').read_text())
    donor_asset=next(x for x in donor_input['objects'] if x['id']=='wooden_spoon')
    signature=lambda a:(a['source_urdf_sha256'],[(x['kind'],x['sha256']) for x in a['meshes']])
    cases=[]
    for ep in episodes:
        name=f'episode_{ep:06d}';inp=B/'episodes'/name;base=OUT/'native_mpc'/name
        manifest=json.loads((inp/'input_manifest.json').read_text());record=json.loads((base/'result.json').read_text())
        assert signature(next(x for x in manifest['objects'] if x['id']=='wooden_spoon'))==signature(donor_asset)
        m=mujoco.MjModel.from_xml_path(record['execution']['scene']);rig=H.HandRig.build(m,'robot_right_','right')
        reference=record['search']['reference'];ref=TR.load_reference(reference,m);original=TR.load_reference(inp/'tracking_reference.npz',m)
        Q=ref['qpos'];tc=m.jnt_qposadr[hand_joints(m,'right')].astype(int);ta=int(m.joint('wooden_spoon_free').qposadr[0])
        assert list(sr.finger_qpos)==list(rig.finger_qpos)
        elevation=original['qpos'][:,ta+2]-original['qpos'][0,ta+2];lift=np.flatnonzero(elevation>.02)
        if not len(lift):raise ValueError('No source spoon lift: needs a separately designed objective')
        lift_start=lift[0]/ref['fps'];lift_end=lift[-1]/ref['fps'];duration=(len(Q)-1)/ref['fps']
        times=dict(start=max(.05,lift_start-1.25),hover=max(.3,lift_start-.85),contact=lift_start-.5,
            closed=lift_start-.2,transport=lift_start+.35,release=min(duration-.35,lift_end+.8))
        for anchor in [2.8,3.5]:
            out=OUT/'pickup_inputs'/name/f'anchor_{anchor:.1f}';out.mkdir(parents=True,exist_ok=True)
            k=round(anchor*float(measured['fps']));d=mujoco.MjData(sm);d.qpos[:]=measured['qpos'][k];mujoco.mj_forward(sm,d)
            objR=Rotation.from_quat(d.qpos[sa+3:sa+7],scalar_first=True)
            localp=objR.inv().apply(d.qpos[sc[:3]]-d.qpos[sa:sa+3])
            localR=objR.inv()*Rotation.from_euler('XYZ',d.qpos[sc[3:6]])
            obj=sm.body('wooden_spoon').id;normals={f:[] for f in ['thumb','index']}
            for c in d.contact:
                b1,b2=int(sm.geom_bodyid[c.geom1]),int(sm.geom_bodyid[c.geom2]);hand=b2 if b1==obj else b1 if b2==obj else -1
                if hand<0:continue
                for f in normals:
                    if f'_{f}_' in sm.body(hand).name:normals[f].append(np.array(c.frame[:3])*(1 if b1==obj else -1))
            if any(not x for x in normals.values()):raise ValueError('Donor anchor lacks actual thumb/index contact')
            before={f:d.site_xpos[sr.fingertip_sites[f]].copy() for f in normals};unit={f:np.mean(ns,axis=0) for f,ns in normals.items()}
            unit={f:n/np.linalg.norm(n) for f,n in unit.items()};targets={f:before[f]+.006*unit[f] for f in normals}
            achieved_closed=d.qpos[list(sr.finger_qpos.values())].copy()
            commanded_closed=np.array([measured['ctrl'][k,a] for a in sr.finger_act.values()])
            H.finger_ik(sm,d,sr,targets,iters=120);opened=d.qpos[list(sr.finger_qpos.values())].copy()
            targetR=Rotation.from_quat(Q[0,ta+3:ta+7],scalar_first=True)
            pose=Q[0].copy();pose[tc[:3]]=Q[0,ta:ta+3]+targetR.apply(localp)
            pose[tc[3:6]]=continuous_euler_xyz(Rotation.from_matrix((targetR*localR).as_matrix()[None]),Q[0,tc[3:6]])[0]
            pose[tc[6:]]=commanded_closed;np.savez_compressed(out/'anchor_pose.npz',qpos=pose,
                measured_closed=achieved_closed,commanded_closed=commanded_closed,local_palm_position=localp,local_palm_rotation=localR.as_matrix())
            spec=dict(episode=name,scene=record['execution']['scene'],reference=reference,
                original_reference=str(inp/'tracking_reference.npz'),source=manifest['source'],geometry=str(inp/'metric_geometry.json'),
                manifest=str(inp/'input_manifest.json'),object_name='wooden_spoon',side='right',anchor_pose=str(out/'anchor_pose.npz'),
                open_fingers=opened.tolist(),finger_names=list(rig.finger_qpos),times=times,
                horizon_s=min(duration,lift_start+2.1),hover_offset=[0.,0.,.06],
                score_hold_start_s=times['closed'],score_track_start_s=lift_start+.35,
                source_donor=str(DONOR),source_anchor_s=anchor,source_recording_sha256=hashlib.sha256((DONOR/'execution/trajectory.npz').read_bytes()).hexdigest(),
                measured_opening_m={f:float((d.site_xpos[sr.fingertip_sites[f]]-before[f])@unit[f]) for f in normals},
                closed_pose_semantics='Actual donor measured palm relative object; actual recorded actuator targets retain finger preload',
                object_targets_changed=False)
            spec['scale']=np.r_[[.008]*3,[.08]*3,[.015]*3,[.08]*3,[.10]*22,[.10]*22,[.010]*3,[.08]*3,[.07,.05,.08]].tolist()
            dump(out/'spec.json',spec);cases.append(dict(episode=ep,anchor=anchor,spec=str(out/'spec.json'),
                output=str(OUT/'pickup_search'/name/f'anchor_{anchor:.1f}')))
    dump(OUT/'pickup_cases.json',cases);print(json.dumps(cases,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--episodes',type=int,nargs='+',default=[37,6,35,36,32,33]);prepare(p.parse_args().episodes)
