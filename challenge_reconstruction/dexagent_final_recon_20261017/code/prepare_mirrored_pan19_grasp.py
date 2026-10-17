"""Mirror a physically achieved right dustpan grip into left-hand proposals.

The reflection is a geometric planning proposal, not a metric symmetry or an
assertion of physical success. Sharpa's mirrored hand uses the same22 measured
finger angles; abduction/adduction signs are not inverted. Full causal physical
execution remains mandatory from the original episode initial state.
"""
from pathlib import Path
import json,hashlib
import numpy as np,mujoco,trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from physics_pickup_search_v2 import B,H,TR,dump
from wrist_reference import continuous_euler_xyz
from lid_grasp_seed import contact_penetration

def main():
    root=B/'refinement_20261018/expanded_batch';donor=root/'hand_margin_policy/episode_000038/evaluation';base=root/'brush19_repaired_pickup_v2/pose_02/search';target=json.loads((base/'result.json').read_text());inp=B/'episodes/episode_000019';out=root/'mirrored_pan19_inputs';out.mkdir(parents=True,exist_ok=True)
    mesh=trimesh.load(B.parent/'final_recon/episode_000019/assets/mini_dust_pan_visual.obj',process=False);points,_=trimesh.sample.sample_surface(mesh,20000,seed=19);center=points.mean(0);_,axes=np.linalg.eigh(np.cov(points.T));tree=cKDTree(points);options=[]
    for normal in axes.T:
        for shift in np.linspace(-.005,.005,21):
            reflected=points-2*np.outer((points-center)@normal-shift,normal);distance=tree.query(reflected)[0]
            options.append((float(np.quantile(distance,.8)),normal.copy(),shift,float(np.median(distance))))
    error,normal,shift,median=min(options,key=lambda x:x[0]);plane_point=center+shift*normal;S=np.eye(3)-2*np.outer(normal,normal);handS=np.diag([1.,-1.,1.])
    sm=mujoco.MjModel.from_xml_path(str(donor/'execution/scene_physics.xml'));sr=H.HandRig.build(sm,'robot_right_','right');record=np.load(donor/'execution/trajectory.npz');anchor=4.5;k=round(anchor*float(record['fps']));d=mujoco.MjData(sm);d.qpos[:]=record['qpos'][k];mujoco.mj_forward(sm,d)
    oc=int(sm.joint('mini_dust_pan_free').qposadr[0]);ob=sm.body('mini_dust_pan').id;wc=np.array([sr.wrist_qpos[x] for x in ['x','y','z','rx','ry','rz']]);R=Rotation.from_quat(d.qpos[oc+3:oc+7],scalar_first=True)
    localp=R.inv().apply(d.qpos[wc[:3]]-d.qpos[oc:oc+3]);localR=R.inv().as_matrix()@Rotation.from_euler('XYZ',d.qpos[wc[3:]]).as_matrix()
    leftp=plane_point+S@(localp-plane_point);leftR=S@localR@handS
    measured=d.qpos[list(sr.finger_qpos.values())].copy();loaded=record['ctrl'][k,list(sr.finger_act.values())].copy();normals={}
    for c in d.contact:
        b1,b2=int(sm.geom_bodyid[c.geom1]),int(sm.geom_bodyid[c.geom2]);hand=b2 if b1==ob else b1 if b2==ob else -1
        if hand not in sr.body_ids or c.dist>0:continue
        finger=next((f for f in H.FINGERS if f'_{f}_' in sm.body(hand).name),None)
        if finger:normals.setdefault(finger,[]).append(S@R.inv().apply(c.frame[:3]*(1 if b1==ob else -1)))
    normals={f:np.mean(v,axis=0) for f,v in normals.items()};normals={f:n/np.linalg.norm(n) for f,n in normals.items()}
    if 'thumb' not in normals or len(normals)<2:raise ValueError('Teacher lacks real thumb/finger geometric contact at selected anchor')
    m=mujoco.MjModel.from_xml_path(target['execution']['scene']);rig=H.HandRig.build(m,'robot_left_','left');ref=TR.load_reference(target['reference'],m);Q=ref['qpos'];tc=int(m.joint('mini_dust_pan_free').qposadr[0]);tw=np.array([rig.wrist_qpos[x] for x in ['x','y','z','rx','ry','rz']]);fc=np.array(list(rig.finger_qpos.values()))
    if list(sr.finger_qpos)!=list(rig.finger_qpos):raise ValueError('Mirrored finger ordering differs')
    rest=np.load(target['execution']['trajectory'])['qpos'][1];targetR=Rotation.from_quat(rest[tc+3:tc+7],scalar_first=True);pose=Q[0].copy();pose[tc:tc+7]=rest[tc:tc+7];pose[tw[:3]]=rest[tc:tc+3]+targetR.apply(leftp);pose[tw[3:]]=continuous_euler_xyz(Rotation.from_matrix((targetR.as_matrix()@leftR)[None]),Q[0,tw[3:]])[0];pose[fc]=loaded
    np.savez_compressed(out/'anchor.npz',qpos=pose,actual_mirrored_fingers=measured,loaded_mirrored_finger_targets=loaded,mirrored_local_palm_position=leftp,mirrored_local_palm_rotation=leftR)
    fk=mujoco.MjData(m);fk.qpos[:]=pose;fk.qpos[fc]=measured;mujoco.mj_forward(m,fk);penetration=contact_penetration(m,fk,rig,m.body('mini_dust_pan').id)
    manifest=json.loads((inp/'input_manifest.json').read_text())
    for i,(opening,hover) in enumerate([(.006,.03),(.010,.03),(.006,.07),(.010,.07)]):
        fk.qpos[:]=pose;fk.qpos[fc]=measured;mujoco.mj_forward(m,fk);targets={f:fk.site_xpos[rig.fingertip_sites[f]].copy()+opening*targetR.apply(n) for f,n in normals.items()};H.finger_ik(m,fk,rig,targets,iters=120)
        times=dict(start=.05,hover=.3,contact=.7,closed=1.,transport=1.6,release=9.45)
        spec=dict(episode='episode_000019',scene=target['execution']['scene'],reference=target['reference'],original_reference=str(inp/'tracking_reference.npz'),source=manifest['source'],manifest=str(inp/'input_manifest.json'),geometry=str(inp/'metric_geometry.json'),
            object_name='mini_dust_pan',side='left',anchor_pose=str(out/'anchor.npz'),open_fingers=fk.qpos[fc].tolist(),times=times,horizon_s=3.4,hover_offset=[0,0,hover],score_hold_start_s=1.,score_track_start_s=1.6,
            original_object_targets_and_clock_unchanged=True,preserved_right_brush_commands=target['reference'],donor=str(donor),donor_anchor_s=anchor,planning_only_mirrored_grasp=True)
        dump(out/f'spec_{i:02d}.json',spec)
    report=dict(source_recording=str(donor/'execution/trajectory.npz'),source_recording_sha256=hashlib.sha256((donor/'execution/trajectory.npz').read_bytes()).hexdigest(),
        donor_anchor_s=anchor,actual_contact_digits=list(normals),reflection_normal_local=normal,reflection_plane_point_local=plane_point,
        approximate_geometry_reflection_p50_m=median,approximate_geometry_reflection_p80_m=error,measured_static_penetration_m=penetration,
        mirror_finger_rule='Same22 angles; no AA sign flip',metric_symmetry_changed=False,physical_success_claim=False)
    dump(out/'planning_report.json',report);print(json.dumps(report,default=lambda x:x.tolist()),flush=True)

if __name__=='__main__':main()
