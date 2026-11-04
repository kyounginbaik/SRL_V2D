"""Episode 25: right-hand grasp proposals on board 2's OVERHANGING end (after the reference slide to the table edge at
8-9 s, as in the video), for physics_pickup_search.py. Unmodified DexAgent grasp-pocket tools; our own board mesh frames.
usage: prepare_board2_pickup.py BASE_RUN OUT_DIR"""
import json, sys
from pathlib import Path
import numpy as np, mujoco, trimesh, yaml
from scipy.spatial.transform import Rotation
LEG=Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017'); sys.path[:0]=['/data1/SRL_V2D/third_party/DexAgent',str(LEG/'code')]
from tool_library.common import hand as H, tracking as TR
from wrist_reference import continuous_euler_xyz
from physics_pickup_search import dump
R7=Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019'); EP=R7/'physics_slots_v3/episodes/episode_000025'
base,out=Path(sys.argv[1]),Path(sys.argv[2]); out.mkdir(parents=True,exist_ok=True)
import os
OBJ=os.environ.get('OBJ','wooden_piece_2'); SIDE=os.environ.get('SIDE','right')
TS=json.loads(os.environ['TS']) if os.environ.get('TS') else dict(start=8.3,hover=8.5,contact=8.8,closed=9.05,transport=9.25,release=15.9); HORIZON=float(os.environ.get('HORIZON','12.5'))
r=json.loads((base/'result.json').read_text()); scene=r['execution']['scene']; man=json.loads((EP/'input_manifest.json').read_text())
m=mujoco.MjModel.from_xml_path(scene); d=mujoco.MjData(m); rig=H.HandRig.build(m,f'robot_{SIDE}_',SIDE)
ref=TR.load_reference(base/'reference.npz',m); Q=ref['qpos']; fps=ref['fps']; k=round(TS['contact']*fps); q=np.load(r['execution']['trajectory'])['qpos'][k].copy()   # simulated state (board where physics put it)
d.qpos[:]=q; mujoco.mj_forward(m,d); ob=m.body(OBJ).id; op=d.xpos[ob].copy(); Rm=d.xmat[ob].reshape(3,3).copy()
A=R7/'collision_recon_slots_v3/episode_000025/assets'; v=np.asarray(trimesh.load(A/f'{OBJ}_visual.obj',force='mesh').vertices); c0=v.mean(0); w,U=np.linalg.eigh((v-c0).T@(v-c0))
if np.linalg.det(U)<0: U[:,0]*=-1
setup=yaml.safe_load(open(EP/'setup/setup.yaml')); tc=np.array(setup['table']['center'][:2])
thin_w,mid_w,long_w=Rm@U[:,0],Rm@U[:,1],Rm@U[:,2]; up=np.array([0,0,1.]); n_up=thin_w*np.sign(thin_w@up)
cands=[]; L=np.ptp((v-c0)@U[:,2])
for s in (1,-1):
    e=op+Rm@c0+s*long_w*L/2; cands.append((np.linalg.norm(e[:2]-tc),s))
s_out=max(cands)[1]; l_out=s_out*long_w
pocket=H.grasp_pocket(m,d,rig,.85); wc=np.array([rig.wrist_qpos[kk] for kk in ('x','y','z','rx','ry','rz')]); fcols=list(rig.finger_qpos.values())
closed=rig.finger_targets(.90); opened=rig.finger_targets(.20)
specs=[]
rad=setup['table']['fit']['radius_m']; Mw=np.ptp((v-c0)@U[:,1]); gl=np.linspace(-L/2+0.005,L/2-0.005,60); gm=np.linspace(-Mw/2+0.005,Mw/2-0.005,30)
G=np.array([c0+U[:,1]*a+U[:,2]*b for a in gm for b in gl]); Gw=G@Rm.T+op; out_d=np.linalg.norm(Gw[:,:2]-tc,axis=1)-rad; ibest=int(np.argmax(out_d)); pstar=G[ibest]
print('furthest board point outside table by %.1f cm'%(out_d[ibest]*100))
for dname,dist in (('c25',0.025),('c40',0.040)):
    dirc=(c0-pstar); dirc[:]=dirc-U[:,0]*(U[:,0]@dirc); dirc/=np.linalg.norm(dirc); point_local=pstar+dist*dirc; point=op+Rm@point_local
    for pname,palm,fdir in (('top_p',-n_up,mid_w),('top_m',-n_up,-mid_w),('end_dn',-l_out,-n_up),('end_mid',-l_out,mid_w)):
        position,euler=H.pocket_pose(pocket,point,palm,fdir); pose=q.copy(); pose[wc[:3]]=position
        pose[wc[3:]]=continuous_euler_xyz(Rotation.from_euler('XYZ',[euler]),q[wc[3:]])[0]; pose[fcols]=[closed[kk] for kk in rig.finger_qpos]
        name=f'{dname}_{pname}'; anchor=out/f'anchor_{name}.npz'; np.savez_compressed(anchor,qpos=pose)
        spec=dict(scene=scene,reference=str(base/'reference.npz'),original_reference=str(EP/'tracking_reference.npz'),source=man['source'],geometry=str(EP/'metric_geometry.json'),
                  manifest=str(EP/'input_manifest.json'),object_name=OBJ,side=SIDE,anchor_pose=str(anchor),open_fingers=[opened[kk] for kk in rig.finger_qpos],
                  finger_names=list(rig.finger_qpos),times=TS,horizon_s=HORIZON,hover_offset=(0.06*(-palm)).tolist(),grasp_point_local=point_local.tolist(),palm_direction_world=palm.tolist(),
                  finger_direction_world=fdir.tolist(),grasp_generator='unmodified DexAgent grasp pocket on the overhanging end of board 2 (our mesh)',previous_teacher=str(base),
                  scale=np.r_[[.012]*3,[.15]*3,[.02]*3,[.12]*3,[.15]*22,[.20]*22,[.015]*3,[.10]*3,[.10,.08,.12]].tolist())
        dump(out/f'{name}.json',spec); specs.append(str(out/f'{name}.json'))
        print(name,'point',np.round(point,3),'outside table by',round((np.linalg.norm(point[:2]-tc)-setup['table']['fit']['radius_m'])*100,1),'cm; palm',np.round(palm,2),'fingers',np.round(fdir,2))
json.dump(specs,open(out/'all_specs.json','w'),indent=1)
