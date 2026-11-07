"""One OakInk clip in CHORD's floating-Sharpa format, with contact variants.
Motion: SPIDER's IK reference for the clip. Contact: official contact_utils (1 cm) on mocap MANO in the object frame, then placed on the reference pose.
Variants: clean = mocap contact. noisy = same, after hand-object offset like video reconstruction (constant + jitter, fixed per clip). none = no contact.
Writes clips/<task>/{object.obj,object_rigid.urdf,<variant>/reference.parquet}."""
import numpy as _n
for _a,_t in dict(bool=bool,int=int,float=float,complex=complex,object=object,str=str,unicode=str,nan=float("nan"),inf=float("inf")).items():
    if _a not in _n.__dict__: setattr(_n,_a,_t)
import sys,json,types,pickle,shutil,numpy as np,torch,trimesh,pyarrow as pa,pyarrow.parquet as pq,xml.etree.ElementTree as ET
from pathlib import Path
from scipy.spatial.transform import Rotation as R
T=sys.argv[1]; NOISE_CM=1.5; NOISE_DEG=10.; JITTER_CM=.3
AB=Path('/scratch2/exaFLOPs/srl_yam_run/ablate'); W=Path('/scratch2/exaFLOPs/spider_wide'); DS=W/'ds/processed/oakinkv2'; RB=Path('/lustre/exaflops/retarget_bench'); SRC=Path('/scratch2/exaFLOPs/SRL_V2D'); OUT=AB/'oak/clips'/T; OUT.mkdir(parents=True,exist_ok=True)
s=(SRC/'robotic_grounding/source/robotic_grounding/robotic_grounding/retarget/params.py').read_text(); b=s[s.index('MANO_HAND_LINKS = {'):]; b=b[:b.index('}')+1]; ns={}; exec(b,ns); LN=ns['MANO_HAND_LINKS']
st=types.ModuleType('robotic_grounding.retarget.params'); st.MANO_HAND_LINKS=LN; st.NUM_MANO_LINKS=len(LN)
for m in ('robotic_grounding','robotic_grounding.retarget'): sys.modules[m]=types.ModuleType(m)
sys.modules['robotic_grounding.retarget.params']=st
cs={}; exec(compile((AB/'video_to_data/reconstruction/modules/v2d_task_library_loader/lib/contact_utils.py').read_text(),'contact_utils','exec'),cs)
ti=json.load(open(DS/f'mano/right/{T}/task_info.json')); tok=ti['oakink_seq_token']; oid=ti['oakink_obj_id']; a0,b0=ti['oakink_window']; dt=float(ti['ref_dt'])
k=np.load(DS/f'sharpa/right/{T}/0/trajectory_kinematic.npz')['qpos']; K0=np.load(DS/f'mano/right/{T}/0/trajectory_keypoints.npz'); oc=K0['qpos_obj_right']; n=len(k)
sh=int(np.argmin([np.abs(k[:,-7:-4]-oc[i:i+n,:3]).max() for i in range(len(oc)-n+1)])); err=float(np.abs(k[:,-7:-4]-oc[sh:sh+n,:3]).max())
an=pickle.load(open(RB/f'data/oakink2_root/anno_preview/{tok}.pkl','rb')); O=an['obj_transf'][oid]; fs=[f for f in range(a0,b0) if f in an['raw_mano'] and f in O]; idx=np.linspace(0,len(fs)-1,len(oc)).round().astype(int)[sh:sh+n]; fs=[fs[i] for i in idx]
M=[an['raw_mano'][f] for f in fs]; pc=torch.cat([m['rh__pose_coeffs'] for m in M]).float(); be=torch.cat([m['rh__betas'] for m in M]).float(); ts=torch.cat([m['rh__tsl'] for m in M]).float()
from manotorch.manolayer import ManoLayer
ml=ManoLayer(mano_assets_root='/scratch2/exaFLOPs/daid_oakink/mano_root',rot_mode='quat',side='right',center_idx=0,use_pca=False,flat_hand_mean=True)
with torch.no_grad(): o=ml(pose_coeffs=pc,betas=be)
V=(o.verts+ts[:,None]).numpy(); J=(o.joints+ts[:,None]).numpy(); faces=ml.th_faces.numpy(); To=np.stack([np.asarray(O[f]) for f in fs])
msrc=Path(str(DS/'assets/objects'))/oid.replace('@','_')/'visual.obj'; shutil.copyfile(msrc,OUT/'object.obj'); mesh=trimesh.load(OUT/'object.obj',force='mesh'); pts,fi=trimesh.sample.sample_surface_even(mesh,4096,seed=0); nor=-mesh.face_normals[fi]; P=torch.tensor(pts,dtype=torch.float32); Nr=torch.tensor(nor,dtype=torch.float32); ids=torch.ones(len(P),dtype=torch.long)
(OUT/'object_rigid.urdf').write_text(f'''<?xml version="1.0" ?>
<robot name="object"><link name="object"><inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="0.3"/><inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial><visual><origin rpy="0 0 0" xyz="0 0 0"/><geometry><mesh filename="object.obj" scale="1 1 1"/></geometry></visual><collision><origin rpy="0 0 0" xyz="0 0 0"/><geometry><mesh filename="object.obj" scale="1 1 1"/></geometry></collision></link></robot>''')
rp=k[:,-7:-4]; rq=k[:,-4:]/np.linalg.norm(k[:,-4:],axis=1,keepdims=True); rr=R.from_quat(rq[:,[1,2,3,0]])
def contact(noisy):
    rng=np.random.default_rng(abs(hash(T))%2**31 if False else sum(map(ord,T)))
    c=np.zeros((n,16,3),np.float32); cn=np.zeros((n,16,3),np.float32); pid=np.zeros((n,16),np.int32)
    dp=rng.normal(size=3); dp*=NOISE_CM/100/np.linalg.norm(dp); dr=rng.normal(size=3); dr*=np.radians(NOISE_DEG)/np.linalg.norm(dr)
    for f in range(n):
        Ti=np.linalg.inv(To[f]); v=V[f]@Ti[:3,:3].T+Ti[:3,3]; j=J[f]@Ti[:3,:3].T+Ti[:3,3]
        if noisy:
            ce=v.mean(0); rot=R.from_rotvec(dr); off=dp+rng.normal(0,JITTER_CM/100,3); v=rot.apply(v-ce)+ce+off; j=rot.apply(j-ce)+ce+off
        vn=trimesh.Trimesh(v,faces,process=False).vertex_normals.copy(); vt=torch.tensor(v,dtype=torch.float32)
        a=cs['approximate_contact_with_id'](P,Nr,ids,vt,torch.tensor(vn,dtype=torch.float32),threshold=.01)
        if len(a[0])==0: continue
        hp,hn,hid=cs['compute_hand_link_contact_positions'](torch.tensor(j,dtype=torch.float32),a[2],a[3],a[4],a[5]); op,on=cs['find_object_contact_positions'](hp,P,Nr); ok=hp.norm(dim=-1)>=1e-3
        o_=op.numpy(); n_=on.numpy(); c[f]=rr[f].apply(o_)+rp[f]; cn[f]=rr[f].apply(n_); pid[f]=(hid*ok).numpy().astype(np.int32)
    c[pid==0]=0; cn[pid==0]=0
    nm=np.linalg.norm(cn,axis=-1,keepdims=True); cn=np.where(nm>1e-6,cn/np.maximum(nm,1e-6),0).astype(np.float32); return c,cn,pid
XMLD=SRC/'robotic_grounding/source/robotic_grounding/robotic_grounding/assets/xmls/sharpawave'
import mujoco
def _hand(side):
    m=mujoco.MjModel.from_xml_path(str(XMLD/f'{side}_sharpawave.xml')); return m,mujoco.MjData(m)
HANDS={}
def urdf_fk(side,q,names):
    """Frames relative to the hand root: site, body or joint anchor by name, from the official Sharpa XML."""
    if side not in HANDS: HANDS[side]=_hand(side)
    m,d=HANDS[side]; d.qpos[:]=0
    for k_,v in q.items():
        j=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,k_); d.qpos[m.jnt_qposadr[j]]=v
    mujoco.mj_kinematics(m,d); root=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,f'{side}_hand_C_MC'); G=np.eye(4); G[:3,:3]=d.xmat[root].reshape(3,3); G[:3,3]=d.xpos[root]; Gi=np.linalg.inv(G); out=[]
    for nm in names:
        M=np.eye(4); s_=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,nm); b_=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,nm); j_=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,nm)
        if s_>=0: M[:3,:3]=d.site_xmat[s_].reshape(3,3); M[:3,3]=d.site_xpos[s_]
        elif b_>=0: M[:3,:3]=d.xmat[b_].reshape(3,3); M[:3,3]=d.xpos[b_]
        elif j_>=0: bid=m.jnt_bodyid[j_]; M[:3,:3]=d.xmat[bid].reshape(3,3); M[:3,3]=d.xanchor[j_]
        elif nm in('universe','root_joint'): M=G.copy()
        else: raise KeyError(nm)
        out.append(Gi@M)
    return np.stack(out)
ref=pq.read_table(AB/'oak/tissue.parquet').to_pylist()[0]
def frames(side,wp,wq,fj,jn,names):
    out=np.zeros((n,len(names),7),np.float32)
    for f in range(0,n):
        G=np.eye(4); G[:3,:3]=R.from_quat(wq[f][[1,2,3,0]]).as_matrix(); G[:3,3]=wp[f]
        Q=urdf_fk(side,dict(zip(jn,fj[f])),names); W_=G@Q; out[f,:,:3]=W_[:,:3,3]; x=R.from_matrix(W_[:,:3,:3]).as_quat(); out[f,:,3:]=x[:,[3,0,1,2]]
    return out
import mujoco
m=mujoco.MjModel.from_xml_path(str(DS/f'sharpa/right/{T}/scene.xml')); d=mujoco.MjData(m); b=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'right_hand_C_MC'); wp=np.zeros((n,3)); wq=np.zeros((n,4))
for f in range(n): d.qpos[:]=k[f]; mujoco.mj_kinematics(m,d); wp[f]=d.xpos[b]; wq[f]=d.xquat[b]
jn=list(ref['right_robot_finger_joint_names']); jm=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,x) for x in jn]; assert min(jm)>=0; fj=np.stack([k[:,m.jnt_qposadr[i]] for i in jm],1)
rf=frames('right',wp,wq,fj,jn,ref['right_robot_frame_names'])
lp=np.tile(rp.mean(0)+[0,.8,.6],(n,1)); lq=np.tile([1.,0,0,0],(n,1)); ln=list(ref['left_robot_finger_joint_names']); lf=np.zeros((n,22)); lfr=frames('left',lp,lq,lf,ln,ref['left_robot_frame_names'])
def write(var,c,cn,pid):
    r=dict(ref); e16=np.zeros((n,16,3),np.float32); e=np.zeros((n,16),np.int32)
    r.update(raw_motion_file=str(tok),fps=1/dt,object_name='object',safe_object_name='object',object_body_names=['object'],safe_object_body_names=['object'],object_mesh_paths=[f'/oak/clips/{T}/object.obj'],object_urdf_paths=[f'/oak/clips/{T}/object_rigid.urdf'],object_mesh_radius=[float(np.linalg.norm(mesh.vertices,axis=1).max())],object_articulation=np.zeros(n).tolist(),object_root_axis_angle=rr.as_rotvec().tolist(),object_root_position=rp.tolist(),object_body_position=rp[:,None].tolist(),object_body_wxyz=rq[:,None].tolist(),
      robot_right_wrist_position=wp.tolist(),robot_right_wrist_wxyz=wq.tolist(),robot_right_finger_joints=fj.tolist(),robot_right_frames=rf.tolist(),robot_left_wrist_position=lp.tolist(),robot_left_wrist_wxyz=lq.tolist(),robot_left_finger_joints=lf.tolist(),robot_left_frames=lfr.tolist(),
      mano_right_object_contact_positions=c.tolist(),mano_right_object_contact_normals=cn.tolist(),mano_right_object_contact_part_ids=pid.tolist(),mano_left_object_contact_positions=e16.tolist(),mano_left_object_contact_normals=e16.tolist(),mano_left_object_contact_part_ids=e.tolist())
    keep=[x for x in r if not x.startswith(('mano_right_tr','mano_right_gl','mano_right_fi','mano_right_jo','mano_right_ti','mano_right_li','mano_left_tr','mano_left_gl','mano_left_fi','mano_left_jo','mano_left_ti','mano_left_li','mano_right_be','mano_left_be','robot_right_frame_task','robot_left_frame_task','robot_right_num','robot_left_num'))]
    r={x:r[x] for x in keep}; (OUT/var).mkdir(exist_ok=True); pq.write_table(pa.Table.from_pylist([r]),OUT/var/'reference.parquet')
cl=contact(False); no=contact(True); write('clean',*cl); write('noisy',*no); write('none',np.zeros((n,16,3),np.float32),np.zeros((n,16,3),np.float32),np.zeros((n,16),np.int32))
ac=(cl[2]>0).any(1); an_=(no[2]>0).any(1); agree=(cl[2]>0)&(no[2]>0); d_=np.linalg.norm(cl[0]-no[0],axis=-1)[agree] if agree.any() else np.array([np.nan])
mv=np.linalg.norm(rp-rp[0],axis=1).max()
q=dict(task=T,frames=n,fps=1/dt,shift=sh,align_err_cm=round(err*100,3),move_cm=round(float(mv*100),1),rise_cm=round(float((rp[:,2]-rp[0,2]).max()*100),1),z0_cm=round(float(rp[0,2]*100),1),mesh_low=round(float(mesh.vertices[:,2].min()*100),1),clean_contact_frames=round(float(ac.mean()),3),noisy_contact_frames=round(float(an_.mean()),3),clean_links=round(float((cl[2]>0).sum(1)[ac].mean()),1) if ac.any() else 0,noisy_links=round(float((no[2]>0).sum(1)[an_].mean()),1) if an_.any() else 0,same_link_frac=round(float(agree.sum()/max((cl[2]>0).sum(),1)),3),same_link_offset_cm=round(float(np.nanmedian(d_)*100),2),wrist_tip_cm=round(float(np.linalg.norm(rf[:,[ref['right_robot_frame_names'].index('right_index_tip_site')],:3]-wp[:,None],axis=-1).mean()*100),1),spider_tip_wrist=round(float(np.linalg.norm(K0['qpos_finger_right'][sh:sh+n,1,:3]-K0['qpos_wrist_right'][sh:sh+n,:3],axis=-1).mean()*100),1))
json.dump(q,open(OUT/'quality.json','w')); print(json.dumps(q))
