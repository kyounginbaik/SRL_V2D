"""Second-hand (left) lateral pinch of the spoon while the right hand keeps its teacher pinch: a two-pinch hold is form-closed
against roll and slide. Left anchor = lateral pinch (thumb/index pads on the two side faces of the handle or bowl, tips at table
level) built by FK from the right hand's own closure mirrored onto the left; closes before the right lifts (1.1 s); transport
follows the immutable object track rigidly. usage: prepare_left_pinch.py BASE_RUN EPISODE_DIR OUT_DIR"""
import json, sys
from pathlib import Path
import numpy as np, mujoco, trimesh
from scipy.spatial.transform import Rotation
LEG=Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017'); sys.path[:0]=['/data1/SRL_V2D/third_party/DexAgent',str(LEG/'code')]
from tool_library.common import hand as H, tracking as TR
from wrist_reference import continuous_euler_xyz
from physics_pickup_search_v2 import dump
base,EP,out=Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]); out.mkdir(parents=True,exist_ok=True)
TS=dict(start=0.3,hover=0.55,contact=0.85,closed=1.05,transport=1.3,release=18.9); HORIZON=8.0
r=json.loads((base/'result.json').read_text()); scene=r['execution']['scene']; man=json.loads((EP/'input_manifest.json').read_text())
m=mujoco.MjModel.from_xml_path(scene); d=mujoco.MjData(m); rl=H.HandRig.build(m,'robot_left_','left'); rr=H.HandRig.build(m,'robot_right_','right')
ref=TR.load_reference(base/'reference.npz',m); Q=ref['qpos']; fps=ref['fps']; k=round(TS['contact']*fps)
q=np.load(r['execution']['trajectory'])['qpos'][k].copy(); d.qpos[:]=q; mujoco.mj_forward(m,d)
ob=m.body('wooden_spoon').id; op=d.xpos[ob].copy(); Rm=d.xmat[ob].reshape(3,3).copy()
mesh=trimesh.load(str(LEG/'episodes/episode_000035/stage2/assets/wooden_spoon/wooden_spoon_visual.obj'),force='mesh')
v=np.asarray(mesh.vertices); c0=v.mean(0); w,U=np.linalg.eigh((v-c0).T@(v-c0)); L=U[:,2]; proj=(v-c0)@L
wid=lambda s: np.ptp(((v-c0)@U[:,1])[np.abs(proj-s*0.12)<0.03])
if wid(1)>wid(-1): L=-L; proj=-proj     # +L toward the handle end
up=np.array([0,0,1.]); lw=Rm@L; lw_h=lw-up*(lw@up); lw_h/=np.linalg.norm(lw_h); side=np.cross(up,lw_h); side/=np.linalg.norm(side)
# left closed config = right's closure at 1.5 s (same joint names, mirrored hand)
closed={kk:float(Q[round(1.5*fps),rr.finger_qpos[kk]]) for kk in rr.finger_qpos}
opened=rl.finger_targets(.20)
# FK of the left pinch in the wrist frame
dd=mujoco.MjData(m); dd.qpos[:]=q; rl.set_fingers(dd,closed); rl.teleport(dd,np.zeros(3),np.zeros(3)); mujoco.mj_forward(m,dd)
T=dd.site_xpos[rl.fingertip_sites['thumb']].copy(); I=dd.site_xpos[rl.fingertip_sites['index']].copy(); M=(T+I)/2; a=(I-T)/np.linalg.norm(I-T); reach=M/np.linalg.norm(M)
print('left pinch: tip gap %.1f cm, midpoint from wrist %.1f cm'%(np.linalg.norm(I-T)*100,np.linalg.norm(M)*100))
wc=np.array([rl.wrist_qpos[kk] for kk in ('x','y','z','rx','ry','rz')]); fcols=list(rl.finger_qpos.values()); specs=[]
for pname,pc in (('h9',0.09),('h12',0.12),('bowl',-0.09)):
    for sname,sgn in (('sA',1.),('sB',-1.)):
        point=op+Rm@(c0+L*pc); a_t=sgn*side; r_t=-up+0.25*sgn*side*0  # tips straight down
        Rw=H.frame_from_axes(a,reach,a_t,r_t); pos=point-Rw@M; pose=q.copy(); pose[wc[:3]]=pos
        pose[wc[3:]]=continuous_euler_xyz(Rotation.from_euler('XYZ',[H.mat_to_euler(Rw)]),q[wc[3:]])[0]; pose[fcols]=[closed[kk] for kk in rl.finger_qpos]
        name=f'leftpinch_{pname}_{sname}'; anchor=out/f'anchor_{name}.npz'; np.savez_compressed(anchor,qpos=pose)
        s=dict(scene=scene,reference=str(base/'reference.npz'),original_reference=str(EP/'tracking_reference.npz'),source=man['source'],geometry=str(EP/'metric_geometry.json'),
               manifest=str(EP/'input_manifest.json'),object_name='wooden_spoon',side='left',anchor_pose=str(anchor),open_fingers=[opened[kk] for kk in rl.finger_qpos],
               finger_names=list(rl.finger_qpos),times=TS,horizon_s=HORIZON,hover_offset=(0.08*up).tolist(),previous_teacher=str(base),grasp_point_local=(c0+L*pc).tolist(),
               grasp_generator='left lateral pinch anchor (right closure mirrored, tips at table level) on the spoon '+pname+'; right-hand teacher pinch unchanged',
               scale=np.r_[[.012]*3,[.15]*3,[.02]*3,[.12]*3,[.15]*22,[.20]*22,[.015]*3,[.10]*3,[.10,.08,.12]].tolist())
        dump(out/f'{name}.json',s); specs.append(str((out/f'{name}.json').resolve()))
        print(name,'wrist pos',np.round(pos,3),'point',np.round(point,3))
json.dump(specs,open(out/'all_specs.json','w'),indent=1); print(len(specs),'specs')
