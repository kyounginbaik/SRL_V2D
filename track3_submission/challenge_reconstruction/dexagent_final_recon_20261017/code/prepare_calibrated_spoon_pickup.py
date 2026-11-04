"""Build collision-aware opening/closing seeds from an observed spoon pinch."""
from pathlib import Path
import json
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from physics_pickup_search_v2 import B, TR, dump, H
from transfer_observed_grasp import hand_joints
from wrist_reference import continuous_euler_xyz


def main():
    W=B/'refinement_20261018/calibrated_scene_candidates'
    inp=W/'episodes/episode_000005';base=W/'runs/episode_000005/temporal50'
    src=B/'refinement_20261018/episode_000003/retention_after45'
    out=W/'pickup_inputs5';out.mkdir(parents=True,exist_ok=True)
    m=mujoco.MjModel.from_xml_path(str(base/'execution/scene_physics.xml'))
    sm=mujoco.MjModel.from_xml_path(str(src/'execution/scene_physics.xml'))
    rig=H.HandRig.build(sm,'robot_right_','right')
    tr=H.HandRig.build(m,'robot_right_','right')
    Q=TR.load_reference(base/'reference.npz',m)['qpos']
    sr=TR.load_reference(src/'reference.npz',sm)
    obs=np.load(src/'execution/trajectory.npz')['qpos']
    sc=sm.jnt_qposadr[hand_joints(sm,'right')].astype(int)
    tc=m.jnt_qposadr[hand_joints(m,'right')].astype(int)
    sa=int(sm.joint('wooden_spoon_free').qposadr[0]);ta=int(m.joint('wooden_spoon_free').qposadr[0])
    for i,(anchor,hover) in enumerate([(2.8,.06),(2.8,.09),(3.5,.06),(3.5,.09)]):
        k=round(anchor*sr['fps']);d=mujoco.MjData(sm);d.qpos[:]=obs[k];mujoco.mj_forward(sm,d)
        ro=Rotation.from_quat(obs[k,sa+3:sa+7],scalar_first=True)
        local_p=ro.inv().apply(obs[k,sc[:3]]-obs[k,sa:sa+3])
        local_R=ro.inv()*Rotation.from_euler('XYZ',obs[k,sc[3:6]])
        obj=sm.body('wooden_spoon').id;normals={f:[] for f in ['thumb','index']}
        for c in d.contact:
            b1,b2=int(sm.geom_bodyid[c.geom1]),int(sm.geom_bodyid[c.geom2])
            hand=b2 if b1==obj else b1 if b2==obj else -1
            if hand<0:continue
            for f in normals:
                if f'_{f}_' in sm.body(hand).name:
                    normals[f].append(np.array(c.frame[:3])*(1 if b1==obj else -1))
        if any(not v for v in normals.values()):raise ValueError('Donor lacks actual thumb/index contact')
        targets={}
        for f,ns in normals.items():
            n=np.mean(ns,axis=0);n/=np.linalg.norm(n)
            targets[f]=d.site_xpos[rig.fingertip_sites[f]].copy()+.006*n
        closed=np.array([sr['qpos'][k,c] for c in rig.finger_qpos.values()])
        H.finger_ik(sm,d,rig,targets,iters=120)
        opened=np.array([d.qpos[c] for c in rig.finger_qpos.values()])
        targetR=Rotation.from_quat(Q[0,ta+3:ta+7],scalar_first=True)
        pose=Q[0].copy();pose[tc[:3]]=Q[0,ta:ta+3]+targetR.apply(local_p)
        pose[tc[3:6]]=continuous_euler_xyz(Rotation.from_matrix((targetR*local_R).as_matrix()[None]),Q[0,tc[3:6]])[0]
        pose[tc[6:]]=closed
        path=out/f'anchor_{i:02d}.npz';np.savez_compressed(path,qpos=pose)
        manifest=json.loads((inp/'input_manifest.json').read_text())
        spec=dict(scene=str(base/'execution/scene_physics.xml'),reference=str(base/'reference.npz'),
            original_reference=str(inp/'tracking_reference.npz'),source=manifest['source'],geometry=str(inp/'metric_geometry.json'),
            manifest=str(inp/'input_manifest.json'),object_name='wooden_spoon',side='right',anchor_pose=str(path),
            open_fingers=opened.tolist(),finger_names=list(tr.finger_qpos),
            times=dict(start=.6,hover=1.0,contact=1.5,closed=1.8,transport=2.5,release=13.55),
            horizon_s=4.,hover_offset=[0.,0.,hover],score_hold_start_s=1.8,score_track_start_s=2.5,
            source_donor=str(src),source_anchor_s=anchor,object_targets_changed=False)
        spec['scale']=np.r_[[.008]*3,[.08]*3,[.015]*3,[.08]*3,[.10]*22,[.10]*22,[.010]*3,[.08]*3,[.07,.05,.08]].tolist()
        dump(out/f'spec_{i:02d}.json',spec)
        print(out/f'spec_{i:02d}.json',flush=True)

if __name__=='__main__':main()
