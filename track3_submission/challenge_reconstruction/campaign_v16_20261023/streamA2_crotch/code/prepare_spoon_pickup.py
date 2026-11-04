"""Spoon pickup-search specs (physics_pickup_search_v2) for a form-closure hold: pocket (power) grasps of the handle from
above (palm down, fingers across the handle, handle point at +3/+6 cm from the centroid toward the handle end, two finger
directions, optional lift of the pocket point) and teacher-anchored pinches with the idle fingers curled. Our mesh, the
teacher's frozen scene, immutable object targets. usage: prepare_spoon_pickup.py BASE_RUN EPISODE_DIR OUT_DIR [release_s]"""
import json, sys
from pathlib import Path
import numpy as np, mujoco, trimesh
from scipy.spatial.transform import Rotation
LEG=Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017'); sys.path[:0]=['/data1/SRL_V2D/third_party/DexAgent',str(LEG/'code')]
from tool_library.common import hand as H, tracking as TR
from wrist_reference import continuous_euler_xyz
from physics_pickup_search_v2 import dump
base,EP,out=Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]); REL=float(sys.argv[4]) if len(sys.argv)>4 else 19.0; out.mkdir(parents=True,exist_ok=True)
TS=dict(start=0.3,hover=0.6,contact=0.95,closed=1.25,transport=1.5,release=REL); HORIZON=8.0
r=json.loads((base/'result.json').read_text()); scene=r['execution']['scene']; man=json.loads((EP/'input_manifest.json').read_text())
m=mujoco.MjModel.from_xml_path(scene); d=mujoco.MjData(m); rig=H.HandRig.build(m,'robot_right_','right')
ref=TR.load_reference(base/'reference.npz',m); Q=ref['qpos']; fps=ref['fps']; k=round(TS['contact']*fps)
q=np.load(r['execution']['trajectory'])['qpos'][k].copy(); d.qpos[:]=q; mujoco.mj_forward(m,d)
ob=m.body('wooden_spoon').id; op=d.xpos[ob].copy(); Rm=d.xmat[ob].reshape(3,3).copy()
mesh=trimesh.load(str(LEG/'episodes/episode_000035/stage2/assets/wooden_spoon/wooden_spoon_visual.obj'),force='mesh')
v=np.asarray(mesh.vertices); c0=v.mean(0); w,U=np.linalg.eigh((v-c0).T@(v-c0)); L=U[:,2]; proj=(v-c0)@L
# +proj end = handle (narrow): verify by width
wid=lambda s: np.ptp(((v-c0)@U[:,1])[np.abs(proj-s*0.12)<0.03]); 
if wid(1)>wid(-1): L=-L; proj=-proj
up=np.array([0,0,1.]); lw=Rm@L; lw_h=lw-up*(lw@up); lw_h/=np.linalg.norm(lw_h); f1=np.cross(up,lw_h); f1/=np.linalg.norm(f1)
pocket=H.grasp_pocket(m,d,rig,.85); wc=np.array([rig.wrist_qpos[kk] for kk in ('x','y','z','rx','ry','rz')]); fcols=list(rig.finger_qpos.values())
opened=rig.finger_targets(.20); specs=[]
def spec_for(name,pose,extra):
    anchor=out/f'anchor_{name}.npz'; np.savez_compressed(anchor,qpos=pose)
    s=dict(scene=scene,reference=str(base/'reference.npz'),original_reference=str(EP/'tracking_reference.npz'),source=man['source'],geometry=str(EP/'metric_geometry.json'),
           manifest=str(EP/'input_manifest.json'),object_name='wooden_spoon',side='right',anchor_pose=str(anchor),open_fingers=[opened[kk] for kk in rig.finger_qpos],
           finger_names=list(rig.finger_qpos),times=TS,horizon_s=HORIZON,previous_teacher=str(base),
           scale=np.r_[[.012]*3,[.15]*3,[.02]*3,[.12]*3,[.15]*22,[.20]*22,[.015]*3,[.10]*3,[.10,.08,.12]].tolist(),**extra)
    dump(out/f'{name}.json',s); specs.append(str(out/f'{name}.json'))
for pc in (0.03,0.06):
    for fname,fdir in (('fA',f1),('fB',-f1)):
        for dz in (0.0,0.015):
            for grip in (0.9,):
                point_local=c0+L*pc; point=op+Rm@point_local+up*dz; palm=-up
                position,euler=H.pocket_pose(pocket,point,palm,fdir); pose=q.copy(); pose[wc[:3]]=position
                pose[wc[3:]]=continuous_euler_xyz(Rotation.from_euler('XYZ',[euler]),q[wc[3:]])[0]; closed=rig.finger_targets(grip); pose[fcols]=[closed[kk] for kk in rig.finger_qpos]
                name=f'pocket_p{int(pc*100)}_{fname}_dz{int(dz*1000)}_g{grip}'
                spec_for(name,pose,dict(hover_offset=(0.08*up).tolist(),grasp_point_local=point_local.tolist(),palm_direction_world=palm.tolist(),finger_direction_world=fdir.tolist(),
                                        grasp_generator='DexAgent grasp pocket (grip .85) on the spoon handle from above, our mesh'))
                print(name,'wrist pos',np.round(position,3),'point z',round(point[2]*100,1),'cm')
# teacher-anchored pinch with idle fingers curled / all fingers closed more
pose=Q[round(1.1*fps)].copy(); pose[fcols]=Q[round(1.5*fps)][fcols]
for name,curl in (('teacher_idlecurl',0.9),('teacher_plain',None)):
    p2=pose.copy()
    if curl is not None:
        for f in ('middle','ring','pinky'):
            for kk in rig.flexion_joints(f): lo,hi=rig.finger_range[kk]; p2[rig.finger_qpos[kk]]=lo+curl*(hi-lo)
    spec_for(name,p2,dict(hover_offset=[0,0,.08],grasp_generator='teacher pinch anchor (wrist at 1.1 s, fingers = 1.5 s closure)'+(' + idle fingers curled' if curl else '')))
json.dump(specs,open(out/'all_specs.json','w'),indent=1); print(len(specs),'specs')
