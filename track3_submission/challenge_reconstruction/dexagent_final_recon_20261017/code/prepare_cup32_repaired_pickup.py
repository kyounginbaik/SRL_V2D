"""Measured cup grasp proposals for a versioned source-supported shape repair."""
from pathlib import Path
import json,hashlib
import mujoco,numpy as np
from scipy.spatial.transform import Rotation
from physics_pickup_search_v2 import B,H,TR,dump
from wrist_reference import continuous_euler_xyz

def main():
    root=B/'refinement_20261018';donor=root/'episode_000010/calibrated_blue_pickup';inp=B/'episodes/episode_000032'
    candidate=root/'expanded_batch/cup32_height_candidate/scale_0.900/original';record=json.loads((candidate/'result.json').read_text());out=root/'expanded_batch/cup32_repaired_pickup_inputs';out.mkdir(parents=True,exist_ok=True)
    sm=mujoco.MjModel.from_xml_path(str(donor/'execution/scene_physics.xml'));sr=H.HandRig.build(sm,'robot_left_','left');z=np.load(donor/'execution/trajectory.npz');k=round(2.5*float(z['fps']));d=mujoco.MjData(sm);d.qpos[:]=z['qpos'][k];mujoco.mj_forward(sm,d)
    oc=int(sm.joint('blue_cup_free').qposadr[0]);wc=np.array([sr.wrist_qpos[x] for x in ['x','y','z','rx','ry','rz']]);body=sm.body('blue_cup').id
    R=Rotation.from_quat(d.qpos[oc+3:oc+7],scalar_first=True);localp=R.inv().apply(d.qpos[wc[:3]]-d.qpos[oc:oc+3]);localR=R.inv()*Rotation.from_euler('XYZ',d.qpos[wc[3:]])
    normals={}
    for c in d.contact:
        b1,b2=int(sm.geom_bodyid[c.geom1]),int(sm.geom_bodyid[c.geom2]);hand=b2 if b1==body else b1 if b2==body else -1
        if hand not in sr.body_ids:continue
        finger=next((f for f in H.FINGERS if f'_{f}_' in sm.body(hand).name),None)
        if finger:normals.setdefault(finger,[]).append(c.frame[:3].copy()*(1 if b1==body else -1))
    normals={f:np.mean(v,axis=0) for f,v in normals.items()};normals={f:n/np.linalg.norm(n) for f,n in normals.items()}
    if 'thumb' not in normals or len(normals)<2:raise ValueError('Donor has no physical opposing contacts')
    m=mujoco.MjModel.from_xml_path(record['execution']['scene']);rig=H.HandRig.build(m,'robot_left_','left');ref=TR.load_reference(inp/'tracking_reference.npz',m);Q=ref['qpos'];tc=int(m.joint('blue_cup_free').qposadr[0]);tw=np.array([rig.wrist_qpos[x] for x in ['x','y','z','rx','ry','rz']])
    rest=np.load(record['execution']['trajectory'])['qpos'][3];rotation=Rotation.from_quat(rest[tc+3:tc+7],scalar_first=True)
    pose=Q[0].copy();pose[tc:tc+7]=rest[tc:tc+7];pose[tw[:3]]=rest[tc:tc+3]+rotation.apply(localp);pose[tw[3:]]=continuous_euler_xyz(Rotation.from_matrix((rotation*localR).as_matrix()[None]),Q[0,tw[3:]])[0]
    measured=z['qpos'][k,list(sr.finger_qpos.values())];loaded=z['ctrl'][k,list(sr.finger_act.values())];pose[list(rig.finger_qpos.values())]=loaded
    np.savez_compressed(out/'anchor.npz',qpos=pose,measured_donor_fingers=measured,loaded_finger_commands=loaded,local_palm_position=localp,local_palm_rotation=localR.as_matrix())
    manifest=json.loads((inp/'input_manifest.json').read_text());duration=(len(Q)-1)/ref['fps']
    for i,(opening,hover) in enumerate([(.003,0.),(.006,.015),(.003,.03)]):
        d.qpos[:]=z['qpos'][k];mujoco.mj_forward(sm,d);targets={f:d.site_xpos[sr.fingertip_sites[f]].copy()+opening*n for f,n in normals.items()};H.finger_ik(sm,d,sr,targets,iters=100)
        times=dict(start=0.,hover=.10,contact=.22,closed=.40,transport=.75,release=min(duration-.35,11.2))
        spec=dict(episode='episode_000032',scene=record['execution']['scene'],reference=str(inp/'tracking_reference.npz'),original_reference=str(inp/'tracking_reference.npz'),
            source=manifest['source'],manifest=str(inp/'input_manifest.json'),geometry=str(inp/'metric_geometry.json'),object_name='blue_cup',side='left',anchor_pose=str(out/'anchor.npz'),
            open_fingers=d.qpos[list(sr.finger_qpos.values())].tolist(),times=times,horizon_s=2.7,hover_offset=[0,0,hover],score_hold_start_s=.4,score_track_start_s=.75,
            object_targets_changed=False,initial_physical_state_unchanged=True,donor=str(donor),donor_anchor_s=2.5,donor_grasp_is_proposal_on_changed_shape=True,
            reconstruction_candidate=str(candidate.parent/'result.json'),source_recording_sha256=hashlib.sha256((donor/'execution/trajectory.npz').read_bytes()).hexdigest())
        dump(out/f'spec_{i:02d}.json',spec)
    print(out,flush=True)

if __name__=='__main__':main()
