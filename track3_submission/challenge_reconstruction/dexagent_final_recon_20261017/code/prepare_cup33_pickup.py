#!/usr/bin/env python3
"""Transfer a physically observed blue-cup grip into episode33 acquisition."""
from pathlib import Path
import json,hashlib
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
from physics_pickup_search_v2 import B,H,TR,dump
from wrist_reference import continuous_euler_xyz


def main():
    root=B/'refinement_20261018';donor=root/'episode_000010/calibrated_blue_pickup'
    donor_input=root/'episode_000010/calibrated_inputs/episodes/episode_000010/input_manifest.json'
    inp=B/'episodes/episode_000033';base=root/'expanded_batch/retention/episode_000033'
    out=root/'expanded_batch/cup33_inputs';out.mkdir(parents=True,exist_ok=True)
    old=json.loads(donor_input.read_text());new=json.loads((inp/'input_manifest.json').read_text())
    asset=lambda x:next(o for o in x['objects'] if o['id']=='blue_cup')
    signature=lambda x:(x['source_urdf_sha256'],[(m['kind'],m['sha256']) for m in x['meshes']])
    if signature(asset(old))!=signature(asset(new)):raise ValueError('Cup asset differs')
    m=mujoco.MjModel.from_xml_path(str(donor/'execution/scene_physics.xml'));rig=H.HandRig.build(m,'robot_left_','left')
    recorded=np.load(donor/'execution/trajectory.npz');k=round(2.5*float(recorded['fps']));d=mujoco.MjData(m);d.qpos[:]=recorded['qpos'][k];mujoco.mj_forward(m,d)
    body=m.body('blue_cup').id;oc=int(m.joint('blue_cup_free').qposadr[0]);wc=np.array([rig.wrist_qpos[key] for key in ['x','y','z','rx','ry','rz']])
    R=Rotation.from_quat(d.qpos[oc+3:oc+7],scalar_first=True);localp=R.inv().apply(d.qpos[wc[:3]]-d.qpos[oc:oc+3]);localR=R.inv()*Rotation.from_euler('XYZ',d.qpos[wc[3:]])
    normals={f:[] for f in H.FINGERS}
    for c in d.contact:
        b1,b2=int(m.geom_bodyid[c.geom1]),int(m.geom_bodyid[c.geom2]);hand=b2 if b1==body else b1 if b2==body else -1
        if hand not in rig.body_ids:continue
        for finger in normals:
            if f'_{finger}_' in m.body(hand).name:normals[finger].append(c.frame[:3].copy()*(1 if b1==body else -1))
    normals={f:np.mean(v,axis=0) for f,v in normals.items() if v};normals={f:n/np.linalg.norm(n) for f,n in normals.items()}
    if 'thumb' not in normals or len(normals)<2:raise ValueError('Donor lacks actual opposing fingers')
    measured=d.qpos[list(rig.finger_qpos.values())].copy();loaded=recorded['ctrl'][k,list(rig.finger_act.values())]
    result=json.loads((base/'result.json').read_text());tm=mujoco.MjModel.from_xml_path(result['execution']['scene']);target=H.HandRig.build(tm,'robot_left_','left')
    Q=TR.load_reference(base/'reference.npz',tm)['qpos'];fps=20.;tc=int(tm.joint('blue_cup_free').qposadr[0]);tw=np.array([target.wrist_qpos[key] for key in ['x','y','z','rx','ry','rz']])
    rotation=Rotation.from_quat(Q[0,tc+3:tc+7],scalar_first=True);pose=Q[0].copy();pose[tw[:3]]=Q[0,tc:tc+3]+rotation.apply(localp)
    pose[tw[3:]]=continuous_euler_xyz(Rotation.from_matrix((rotation*localR).as_matrix()[None]),Q[0,tw[3:]])[0];pose[list(target.finger_qpos.values())]=loaded
    np.savez_compressed(out/'observed_anchor.npz',qpos=pose,measured_fingers=measured,loaded_finger_targets=loaded,
        donor_local_palm_position=localp,donor_local_palm_rotation=localR.as_matrix())
    lifted=np.flatnonzero(Q[:,tc+2]-Q[0,tc+2]>.02);start=lifted[0]/fps;end=lifted[-1]/fps
    times=dict(start=max(.05,start-1.25),hover=max(.3,start-.85),contact=start-.5,closed=start-.2,transport=start+.4,release=min((len(Q)-1)/fps-.35,end+.7))
    for i,(opening,hover) in enumerate([(.006,.07),(.010,.07),(.006,.10),(.010,.10)]):
        d.qpos[:]=recorded['qpos'][k];mujoco.mj_forward(m,d)
        targets={f:d.site_xpos[rig.fingertip_sites[f]].copy()+opening*n for f,n in normals.items()};H.finger_ik(m,d,rig,targets,iters=120)
        spec=dict(episode='episode_000033',scene=result['execution']['scene'],reference=str(base/'reference.npz'),
            original_reference=str(inp/'tracking_reference.npz'),source=new['source'],manifest=str(inp/'input_manifest.json'),geometry=str(inp/'metric_geometry.json'),
            object_name='blue_cup',side='left',anchor_pose=str(out/'observed_anchor.npz'),open_fingers=d.qpos[list(rig.finger_qpos.values())].tolist(),
            times=times,horizon_s=start+2.2,hover_offset=[0,0,hover],score_hold_start_s=times['closed'],score_track_start_s=times['transport'],
            object_targets_changed=False,donor=str(donor),donor_anchor_s=2.5,donor_contacts=list(normals),
            source_recording_sha256=hashlib.sha256((donor/'execution/trajectory.npz').read_bytes()).hexdigest())
        dump(out/f'spec_{i:02d}.json',spec)
    print(json.dumps(dict(output=str(out),times=times,donor_contacts=list(normals))))


if __name__=='__main__':main()
