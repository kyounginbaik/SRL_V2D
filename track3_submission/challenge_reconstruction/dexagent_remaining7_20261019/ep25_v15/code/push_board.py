"""Episode 25: robot push primitive that slides a flat board outward to overhang the table edge (as the person does
before lifting). Hand comes down palm-down over the board's top centre (DexAgent grasp pocket pose), fingers half closed so the
pads touch the top face, presses PRESS m, slides DIST m horizontally away from the table centre over [TC, TE], then rises.
Only the chosen hand's commands in [TC-0.35, TE+0.25] are replaced; executes and reports the board overhang over time.
usage: push_board.py SRC_RUN OUT SIDE OBJECT TC TE DIST PRESS"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import sys, json
from pathlib import Path
import numpy as np, mujoco, trimesh, yaml
from scipy.spatial.transform import Rotation
LEG=Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017'); sys.path[:0]=['/data1/SRL_V2D/third_party/DexAgent',str(LEG/'code')]
from tool_library.common import hand as H, tracking as TR
from wrist_reference import continuous_euler_xyz
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files
R7=Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019'); EP=R7/'physics_slots_v3/episodes/episode_000025'
src,out,side,obj=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3],sys.argv[4]; tc,te,dist,press=map(float,sys.argv[5:9])
prev=json.loads((src/'result.json').read_text()); scene=prev['execution']['scene']; m=mujoco.MjModel.from_xml_path(scene); d=mujoco.MjData(m)
ref=TR.load_reference(src/'reference.npz',m); Q=ref['qpos'].copy(); fps=ref['fps']; t=np.arange(len(Q))/fps
traj=np.load(prev['execution']['trajectory']); k0=round((tc-0.35)*fps); d.qpos[:]=traj['qpos'][k0]; mujoco.mj_forward(m,d)
ob=m.body(obj).id; op=d.xpos[ob].copy(); Rm=d.xmat[ob].reshape(3,3).copy()
A=R7/'collision_recon_slots_v3/episode_000025/assets'; v=np.asarray(trimesh.load(A/f'{obj}_visual.obj',force='mesh').vertices); c0=v.mean(0); w,U=np.linalg.eigh((v-c0).T@(v-c0))
setup=yaml.safe_load(open(EP/'setup/setup.yaml')); tcen=np.array(setup['table']['center'][:2]); rad=setup['table']['fit']['radius_m']
centre=op+Rm@c0; topz=(v@Rm.T+op)[:,2].max(); mid_w=Rm@U[:,1]
out_dir=np.r_[centre[:2]-tcen,0]; out_dir/=np.linalg.norm(out_dir)
rig=H.HandRig.build(m,f'robot_{side}_',side); pocket=H.grasp_pocket(m,d,rig,.85)
wc=np.array([rig.wrist_qpos[kk] for kk in ('x','y','z','rx','ry','rz')]); fcols=list(rig.finger_qpos.values()); half=rig.finger_targets(.45)
MODE=os.environ.get('PUSH_MODE','edge')
if MODE=='top':
    point=np.r_[centre[:2],topz]; position,euler=H.pocket_pose(pocket,point,np.array([0,0,-1.]),mid_w)
else:
    X=v@Rm.T+op; proj=(X[:,:2]-centre[:2])@out_dir[:2]; inner=centre[:2]+out_dir[:2]*proj.min()   # inner edge (towards the table centre)
    point=np.r_[inner-out_dir[:2]*float(os.environ.get('EDGE_BACK','0.015')),centre[2]+float(os.environ.get('EDGE_DZ','0.01'))]
    position,euler=H.pocket_pose(pocket,point,out_dir,np.array([0,0,-1.]))
eul=continuous_euler_xyz(Rotation.from_euler('XYZ',[euler]),Q[k0][wc[3:]])[0]
b=lambda x: np.clip(x,0,1)**3*(10-15*np.clip(x,0,1)+6*np.clip(x,0,1)**2)
ks=[i for i in range(len(Q)) if tc-0.35<=t[i]<=te+0.25]; base=Q.copy()
for i in ks:
    s=t[i]; down=b((s-(tc-0.35))/0.3); slide=b((s-tc)/(te-tc)); up=b((s-te)/0.2)
    p=position+np.r_[0,0,0.04]*(1-down)+np.r_[0,0,-press]*down*(1-up)*(MODE=='top')+out_dir*dist*slide+np.r_[0,0,0.05]*up
    blend=b((s-(tc-0.35))/0.15)*(1-b((s-(te+0.1))/0.15))
    Q[i,wc[:3]]=(1-blend)*base[i,wc[:3]]+blend*p; Q[i,wc[3:]]=(1-blend)*base[i,wc[3:]]+blend*eul
    Q[i,fcols]=(1-blend)*base[i,fcols]+blend*np.array([half[kk] for kk in rig.finger_qpos])
out.mkdir(parents=True,exist_ok=True); refp=TR.save_reference(out/'reference.npz',m,Q,ref['labels'],fps,ref['segment'],ref['segments'],source=f'push {obj} {dist} m outward [{tc},{te}] press {press}',floating_runtime=True,no_retiming=True)
man=json.loads((EP/'input_manifest.json').read_text()); ex=execute_reference(scene,refp,out/'execution'); met=evaluate_files(man['source'],ex['trajectory'],EP/'input_manifest.json',ex['scene'],EP/'metric_geometry.json')
a=np.load(ex['trajectory']); j=list(a['object_names']).index(obj); ovh=[]
for s in (tc,te,te+0.3,te+0.6):
    kk=min(round(s*fps),len(a['object_positions'])-1); Rq=Rotation.from_quat(a['object_quaternions_wxyz'][kk,j][[1,2,3,0]]).as_matrix(); X=v@Rq.T+a['object_positions'][kk,j]
    ovh.append(round(float((np.linalg.norm(X[:,:2]-tcen,axis=1).max()-rad)*100),1))
res=dict(episode='episode_000025',input_dir=str(EP),reference=str(refp),execution=ex,metrics=met,parent=str(src),push=dict(side=side,obj=obj,tc=tc,te=te,dist=dist,press=press,overhang_cm=ovh),spec=prev.get('spec'))
(out/'result.json').write_text(json.dumps(res,indent=1,default=lambda x: x.tolist() if isinstance(x,np.ndarray) else str(x)))
print(json.dumps(dict(out=out.name,overhang_cm_at_tc_te_te3_te6=ovh,objects={kk:(round(vv['mean_position_error_cm'][0],2),round(vv['mean_orientation_error_degrees'][0],1)) for kk,vv in met['official_style_reference_proxy']['objects'].items()})),flush=True)
