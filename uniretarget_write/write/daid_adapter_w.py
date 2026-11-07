"""Write one OakInk-v2 clip in the format Do As I Do's reconstruction pipeline outputs, so their retargeting runs unchanged.
Same sequence, object and 4 s window as SPIDER's clip, sampled at 30 fps (their ref_dt). Run with .venv_spider."""
import numpy as _n
for _a,_t in dict(bool=bool,int=int,float=float,complex=complex,object=object,str=str,unicode=str,nan=float("nan"),inf=float("inf")).items():
    if _a not in _n.__dict__: setattr(_n,_a,_t)
import sys,os,json,pickle,shutil,numpy as np,torch
from scipy.spatial.transform import Rotation as R
from manotorch.manolayer import ManoLayer
T,out=sys.argv[1:3]
BR='/lustre/exaflops/retarget_bench'; DD='/scratch2/exaFLOPs/spider_wide/ds/processed/oakinkv2'
ti=json.load(open(f'{DD}/mano/right/{T}/task_info.json')); tok=ti['oakink_seq_token']; oid=ti['oakink_obj_id']; a,b=ti['oakink_window']
anno=pickle.load(open(f'{BR}/data/oakink2_root/anno_preview/{tok}.pkl','rb'))
fids=[f for f in range(a,b) if f in anno['raw_mano'] and f in anno['obj_transf'][oid]][::4]
N=len(fids); M=[anno['raw_mano'][f] for f in fids]
pc=torch.cat([m['rh__pose_coeffs'] for m in M]).float(); be=torch.cat([m['rh__betas'] for m in M]).float(); ts=torch.cat([m['rh__tsl'] for m in M]).float()
ml=ManoLayer(mano_assets_root='/scratch2/exaFLOPs/daid_oakink/mano_root',rot_mode='quat',side='right',center_idx=0,use_pca=False,flat_hand_mean=True)
with torch.no_grad(): o=ml(pose_coeffs=pc,betas=be)
Z=np.array([0,0,1.0])  # their loader requires positive depth; a constant shift, removed again by their recentring
J=(o.joints+ts[:,None]).numpy().astype(np.float64)+Z; V=(o.verts+ts[:,None]).numpy().astype(np.float64)+Z
q=pc.numpy(); rv=lambda x:R.from_quat(x[...,[1,2,3,0]]).as_rotvec()
os.makedirs(f'{out}/{T}',exist_ok=True)
np.savez(f'{out}/{T}/all_hand_meshes.npz',right_joints=J,right_valid=np.ones(N,bool),right_rot=rv(q[:,0]),right_vertices=V,right_faces=ml.th_faces.numpy(),right_hand_pose=rv(q[:,1:]).reshape(N,45),right_betas=be.numpy())
name=oid.replace('@','_'); json.dump({'object_names':[name],'anchor_hand':'right'},open(f'{out}/config.json','w'))
json.dump({'vec3d':[0,1,0],'roll_deg':0,'pitch_deg':0},open(f'{out}/gravity.json','w'))
objs=[]
for i,f in enumerate(fids):
    X=np.asarray(anno['obj_transf'][oid][f]); x=R.from_matrix(X[:3,:3]).as_quat()
    objs.append({'frame_idx':i,'local_to_scene':{'translation_camera_frame':(X[:3,3]+Z).tolist(),'quat_wxyz_camera_frame':[x[3],x[0],x[1],x[2]]}})
L=f'{out}/obj_tracking_out/{name}/combined_visualization'; os.makedirs(L,exist_ok=True)
json.dump({'objects':objs,'translation_scale_optimization':{'mesh_scale':1.0}},open(f'{L}/layout_camera_frame_optimized.json','w'))
m=f'{out}/video_segmentation/masks/frame_0000_masks/{name}'; os.makedirs(m,exist_ok=True); shutil.copyfile(ti['right_object_mesh_dir'].replace('processed/oakinkv2',DD)+'/visual.obj',f'{m}/{name}.obj')
# Sanity against SPIDER's clip of the same data, including height, which is what caught the wrong up axis.
k=np.load(f'{DD}/mano/right/{T}/0/trajectory_keypoints.npz'); idx=np.linspace(0,len(k['qpos_obj_right'])-1,N).round().astype(int)
so=k['qpos_obj_right'][idx,:3]; me=np.stack([np.asarray(anno['obj_transf'][oid][f])[:3,3] for f in fids])
sw=k['qpos_wrist_right'][idx,:3]; st=k['qpos_finger_right'][idx,:,:3]
print(json.dumps(dict(task=T,frames=N,obj_path_cm=round(float(np.linalg.norm(me-me[0],axis=1).max()*100),1),spider_obj_path_cm=round(float(np.linalg.norm(so-so[0],axis=1).max()*100),1),tip_wrist_cm=round(float(np.linalg.norm(J[:,[4,8,12,16,20]]-J[:,:1],axis=-1).mean()*100),2),spider_tip_wrist_cm=round(float(np.linalg.norm(st-sw[:,None],axis=-1).mean()*100),2),obj_rise_cm=round(float(np.ptp(me[:,1])*100),1),spider_rise_cm=round(float(np.ptp(so[:,2])*100),1),tip_obj_min_cm=round(float(np.linalg.norm(J[:,[4,8,12,16,20]]-Z-me[:,None],axis=-1).min()*100),1))))
