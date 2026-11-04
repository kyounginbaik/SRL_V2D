#!/usr/bin/env python3
"""Transfer verified HaWoR camera-space tracks into the supplied object world.

Three independent phases: prepare provenance; decode MANO on CPU in the isolated
MANO image; exact Sharpa Wave Mink IK in the existing retarget image. The input
ZIP and extracted original bundle are never modified. No physics is certified.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, shutil, subprocess, sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
BASE=Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019')
ROOT=Path('/data1/SRL_V2D')
EPISODES=[14,15,16,20,25,28,30]
OUTROOT=Path(os.environ.get('HAWOR_OUTPUT_ROOT',str(BASE/'hawor_recon')))
HAWOR=ROOT/'challenge_submissions/track3/eval_hawor_latest_20261008/outputs'
OLD_MINK=ROOT/'challenge_reconstruction/eval_mink_ready_20261010/code/run_mink.py'
MANO=ROOT/'reconstruction/data/weights/wilor/pretrained_models'
LOADER=ROOT/'reconstruction/modules/v2d_task_library_loader/lib/read_mano.py'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n')
def load(p):return json.loads(Path(p).read_text())
def arr(x):
 if isinstance(x,np.ndarray) and x.dtype==object:return np.asarray([arr(v) for v in x])
 return np.asarray(x)
def choose(e):
 candidates=[]
 for p in sorted((HAWOR/f'episode_{e:06d}').iterdir()):
  if not (p/'hawor/_metadata.json').is_file():continue
  meta=load(p/'hawor/_metadata.json')
  assert meta['source']=='hawor'
  sides={s:{} for s in ['left','right']}
  for f in sorted((p/'hawor_aligned_filled').glob('*/*.json')):
   if f.parent.name not in [str(meta['left_id']),str(meta['right_id'])]:continue
   d=load(f);side='right' if d['is_right'] else 'left'
   assert str(d['track_id'])==str(meta[side+'_id']),(f,meta)
   raw=p/'hawor'/f.parent.name/f.name
   if raw.exists() and not d.get('interpolated',False):
    rd=load(raw)
    assert np.allclose(rd['mano']['hand_pose'],d['mano']['hand_pose'],atol=1e-7),(f,'stale non-HaWoR finger pose')
    assert np.allclose(rd['mano']['global_orient'],d['mano']['global_orient'],atol=1e-7),(f,'stale non-HaWoR wrist rotation')
   if d['frame_idx'] in sides[side]:raise ValueError(f'duplicate hand {f}')
   sides[side][d['frame_idx']]=(d,f)
  if all(sides.values()):candidates.append((min(map(len,sides.values())),sum(map(len,sides.values())),str(p),sides,meta))
 if not candidates:raise ValueError(f'No verified bilateral HaWoR for {e}')
 _,_,p,sides,meta=max(candidates,key=lambda x:x[:3]);return Path(p),sides,meta

def prepare(e):
 import pyarrow.parquet as pq, trimesh
 src=BASE/f'source/final_recon_ours/episode_{e:06d}'
 out=OUTROOT/f'episode_{e:06d}'
 if out.exists():
  if (out/'hawor_prepared.json').is_file():return
  raise RuntimeError(f'Partial existing destination requires inspection: {out}')
 shutil.copytree(src,out)
 # Original WiLoR data and original videos are archival; they must not masquerade
 # as the newly generated hand reference.
 archive=out/'archive_wilor';archive.mkdir()
 for name in ['reference.parquet','reference.mp4','ego_comparison.mp4','hands_stereo.npz','hand_targets.npz','hand_validity.npz','sharpa_ik.npz','hand_report.json','validation.json','USAGE.txt']:
  p=out/name
  if p.exists():shutil.move(p,archive/name)
 c=load(src/'config.json');N=int(c['frames']);cam=np.load(src/'camera_motion.npz')
 assert int(cam['fps'])==20 and np.array_equal(cam['frames'],np.arange(N))
 W=cam['world_from_rgb'];assert W.shape==(N,4,4)
 assert np.max(np.abs(np.linalg.det(W[:,:3,:3])-1))<1e-5
 p,sides,meta=choose(e)
 assert meta['source_frame_count']==N,(e,meta,N)
 records={'source':str(p),'source_metadata':meta,'frames':N,'fps':20,'sides':{}}
 for side,items in sides.items():
  assert min(items)>=0 and max(items)<N
  payload=[]
  for f,(d,path) in sorted(items.items()):
   d=dict(d);d['_source_file']=str(path);d['_source_sha256']=sha(path);payload.append(d)
  records['sides'][side]=payload
 dump(out/'hawor_tracks.json',records)
 # Relocate asset references in URDF only when present. The ZIP's mesh bytes,
 # object trajectories and body coordinate systems remain untouched.
 relocated=[]
 for f in (out/'assets').glob('*.urdf'):
  s=f.read_text();new=s.replace('/ndata/jisu/final_recon_ours',str(BASE/'hawor_recon'))
  if s!=new:f.write_text(new);relocated.append(f.name)
 np.savez_compressed(out/'hawor_camera_transform.npz',world_from_rgb=W,K=cam['K'],frames=cam['frames'],fps=20.)
 official=load(ROOT/'challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261017/submission_readiness/latest_official/v2d_submission_kit/data/track_3_evaluation_objects.json')['episodes'][str(e)]
 poses=np.load(src/'object_poses.npz');assert list(poses['names'])==official
 audit={'episode':e,'source_zip':str(ROOT/'final_recon_ours.zip'),'input_hand_source':'WiLoR (replaced by HaWoR)','frames':N,'fps':20,'duration_last_frame_seconds':(N-1)/20,'object_order':official,'official_object_order_matches':True,'source_objects_preserved':True,'input_reference_sha256':sha(src/'reference.parquet'),'source_camera_sha256':sha(src/'camera_motion.npz'),'source_poses_sha256':sha(src/'object_poses.npz'),'hawor_source':str(p),'hawor_metadata':meta,'urdfs_relocated':relocated,'assets':{},'notes':['Input collision decomposition is approximate k-means hulls; grasp physics must be evaluated.','Input Sharpa sites used parent-body approximations; all robot FK is recomputed using exact Sharpa XML.','Camera axes unchanged by standard undistortion. ZIP world_from_rgb is interpreted in the same OpenCV RGB camera frame; different intrinsic matrices affect projection, not the 3D rigid transform.']}
 for obj in official:
  m=trimesh.load(src/f'assets/{obj}.obj',force='mesh',process=False)
  audit['assets'][obj]={'sha256':sha(src/f'assets/{obj}.obj'),'vertices':len(m.vertices),'faces':len(m.faces),'extents_m':m.extents.tolist(),'finite':bool(np.isfinite(m.vertices).all())}
 dump(out/'hawor_prepared.json',audit)
 print('prepared',e,flush=True)

def decode(e):
 import torch
 torch.set_num_threads(2)
 # MANO GPL dependency stays inside its dedicated existing image.
 spec=importlib.util.spec_from_file_location('local_read_mano',LOADER);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 mano=mod.MANO(str(MANO),device=torch.device('cpu'),flat_hand_mean=True,center_idx=None)
 out=OUTROOT/f'episode_{e:06d}';tracks=load(out/'hawor_tracks.json');N=tracks['frames']
 C=np.load(out/'camera_motion.npz')['world_from_rgb'].astype(float);rc=C[:,:3,:3];tc=C[:,:3,3]
 data={};stats={}
 for side in ['left','right']:
  records=tracks['sides'][side];ix=np.array([x['frame_idx'] for x in records],int)
  betas=np.array([x['mano']['betas'] for x in records],float)
  aa=np.array([x['mano']['global_orient'] for x in records],float)
  fp=np.array([x['mano']['hand_pose'] for x in records],float)
  trans=np.array([x['camera']['cam_t'] for x in records],float)
  scale=np.array([x.get('hand_scale',1.) for x in records],float)
  # JSONs are normalized to the right-MANO + mirror-X renderer contract.
  # Convert exactly once back to native left-MANO axis-angle coordinates.
  if side=='left':
   aa[:,1:]*=-1;fp=fp.reshape(-1,15,3);fp[:,:,1:]*=-1;fp=fp.reshape(-1,45)
  tens=lambda x:torch.as_tensor(x,dtype=torch.float32)
  with torch.no_grad():
   result=mano.forward(side,tens(betas),tens(aa),torch.zeros((len(ix),3)),tens(fp))
  vertices=result['vertices'].numpy().astype(float);joints=result['joints'].numpy().astype(float)
  centroid=vertices.mean(axis=1,keepdims=True)
  # Existing depth-aligned HaWoR renderer scales about the vertex centroid.
  vertices=(vertices-centroid)*scale[:,None,None]+centroid+trans[:,None,:]
  joints=(joints-centroid)*scale[:,None,None]+centroid+trans[:,None,:]
  q=result['joints_wxyz'].numpy();qrot=R.from_quat(q.reshape(-1,4)[:,[1,2,3,0]]).as_matrix().reshape(len(ix),21,3,3)
  worldj=np.einsum('nij,nkj->nki',rc[ix],joints)+tc[ix,None,:]
  worldv=np.einsum('nij,nkj->nki',rc[ix],vertices)+tc[ix,None,:]
  worldr=np.einsum('nij,nkjl->nkil',rc[ix],qrot)
  worldq=R.from_matrix(worldr.reshape(-1,3,3)).as_quat(scalar_first=True).reshape(len(ix),21,4)
  # Gaps are filled in WORLD coordinates to prevent unseen parked hands following
  # head motion. They remain explicitly invalid for observed-hand supervision.
  fullix=np.arange(N)
  def interp(v):
   flat=v.reshape(len(ix),-1)
   return np.stack([np.interp(fullix,ix,flat[:,j]) for j in range(flat.shape[1])],1).reshape((N,)+v.shape[1:])
  jfull=interp(worldj);vfull=interp(worldv)
  qfull=np.empty((N,21,4))
  for j in range(21):qfull[:,j]=Slerp(ix,R.from_quat(worldq[:,j,[1,2,3,0]]))(np.clip(fullix,ix[0],ix[-1])).as_quat(scalar_first=True)
  valid=np.zeros(N,bool);interpolated=np.ones(N,bool);available=np.zeros(N,bool)
  valid[ix]=[not x.get('interpolated',False) for x in records]
  interpolated[ix]=[x.get('interpolated',False) for x in records];available[ix]=True
  # Parameter columns are descriptive; actual scaled FK targets are the authoritative
  # joints/vertices used by IK. Store scale separately, never silently discard it.
  global_world=R.from_matrix(np.einsum('nij,njk->nik',rc[ix],R.from_rotvec(aa).as_matrix())).as_rotvec()
  for key,val in [('joints_world',jfull),('vertices_world',vfull),('joints_wxyz',qfull),('betas',np.median(betas,axis=0)),('finger_pose',interp(fp)),('global_orient',interp(global_world)),('valid',valid),('interpolated',interpolated),('available',available),('hand_scale',interp(scale[:,None])[:,0])]:data[side+'_'+key]=val
  invproj=np.einsum('nji,nkj->nki',rc[ix],jfull[ix]-tc[ix,None,:])
  roundtrip=float(np.max(np.abs(invproj-joints)));assert roundtrip<1e-6
  assert np.isfinite(jfull).all() and np.isfinite(qfull).all()
  stats[side]={'observed_frames':int(valid.sum()),'source_filled_frames':int(len(ix)-valid.sum()),'world_interpolated_or_endpoint_held_frames':int((~available).sum()),'missing_source_frames':fullix[~available].tolist(),'source_frame_range':[int(ix[0]),int(ix[-1])],'depth_scale_range':[float(scale.min()),float(scale.max())],'camera_world_roundtrip_max_abs_m':roundtrip}
 data['fps']=np.array(20.);data['frames']=np.arange(N)
 np.savez_compressed(out/'hawor_decoded.npz',**data)
 dump(out/'hawor_decode_report.json',{'hand_source':'HaWoR','source':tracks['source'],'mano_decoder':str(LOADER),'mano_decoder_sha256':sha(LOADER),'left_conversion':'right-parameter JSON to native-left MANO: negate axis-angle yz once','flat_hand_mean':True,'scale_contract':'depth scale applied about MANO vertex centroid before camera translation','gap_filling':'world-space linear positions and SLERP orientations, endpoint hold; all filled frames flagged','sides':stats,'object_poses_modified':False})
 print('decoded',e,stats,flush=True)

def ik(e):
 import pyarrow as pa, pyarrow.parquet as pq
 import torch
 from scipy.signal import savgol_filter
 torch.set_num_threads(2)
 modpath=BASE/'code/mink_sharpa_exact.py';spec=importlib.util.spec_from_file_location('mink_sharpa_exact',modpath);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 from robotic_grounding.retarget import SHARPA_WAVE_XMLS_DIR
 out=OUTROOT/f'episode_{e:06d}';src=BASE/f'source/final_recon_ours/episode_{e:06d}'
 table=pq.ParquetFile(src/'reference.parquet').read();row=table.to_pylist()[0]
 h=np.load(out/'hawor_decoded.npz');N=len(h['frames']);data={};report={'hand_source':'HaWoR','solver':'exact Sharpa Wave floating 22-finger-DOF + 6 wrist-DOF per hand, Mink 1.3','source':'hawor_decoded.npz','sides':{},'objects_preserved':True,'physics_validated':False}
 handj=[];handr=[]
 for side in ['left','right']:
  obj=mod.MinkSharpa(side=side,robot_asset_path=str(SHARPA_WAVE_XMLS_DIR/f'{side}_sharpawave.xml'),source_model='mano',max_iter=80,frequency=200.)
  j=h[side+'_joints_world'].copy();q=h[side+'_joints_wxyz'].copy();w=9
  for t in range(1,N):q[t][(q[t]*q[t-1]).sum(-1)<0]*=-1
  j=savgol_filter(j,w,2,axis=0);q=savgol_filter(q,w,2,axis=0);q/=np.linalg.norm(q,axis=-1,keepdims=True)
  results=[];prev=None
  for t in range(N):
   if prev is None:
    prev=obj.robot.q0.copy();prev[:3]=j[t,0];prev[3:7]=q[t,0,[1,2,3,0]]
   result=obj.compute(j[t],q[t],source_to_robot_scale=1.,qpos=prev);prev=result['q'];results.append(result)
   if t%50==0:print('IK',e,side,t,flush=True)
  pose=np.array([r['q'] for r in results]);raw=pose.copy()
  for t in range(1,N):
   if np.dot(pose[t,3:7],pose[t-1,3:7])<0:pose[t,3:7]*=-1
  pose=savgol_filter(pose,11,2,axis=0);pose[:,3:7]/=np.linalg.norm(pose[:,3:7],axis=1,keepdims=True)
  pose[:,7:]=np.clip(pose[:,7:],obj.robot.model.lowerPositionLimit[7:],obj.robot.model.upperPositionLimit[7:])
  frames=[];errors=[]
  for t,qt in enumerate(pose):
   obj.configuration.update(qt);obj.mc.update(obj.to_mj(qt));f=[];err=[]
   for name in obj.robot_frame_names.values():
    T=obj.configuration.get_transform_frame_to_world(name);f.append(np.r_[T.translation,R.from_matrix(T.rotation).as_quat(scalar_first=True)])
   for name,task in obj.mt.items():
    rot,pos=obj.targets[t][name];task.set_target(mod.mink.SE3.from_rotation_and_translation(mod.mink.SO3.from_matrix(rot),pos));err.append(float(np.linalg.norm(task.compute_error(obj.mc)[:3])))
   frames.append(f);errors.append(err)
  prefix='robot_'+side;mp='mano_'+side
  row[prefix+'_wrist_position']=pose[:,:3].tolist();row[prefix+'_wrist_wxyz']=pose[:,[6,3,4,5]].tolist();row[prefix+'_finger_joints']=pose[:,7:].tolist()
  row[prefix+'_frames']=np.array(frames).tolist();row[prefix+'_frame_task_errors']=errors;row[prefix+'_num_optimization_iterations']=[r['num_optimization_iterations'] for r in results]
  row[side+'_robot_frame_names']=list(obj.robot_frame_names.values());row[side+'_robot_finger_joint_names']=list(obj.robot_finger_joint_names.values());row[side+'_robot_frame_task_names']=list(obj.mt)
  row[mp+'_betas']=h[side+'_betas'].tolist();row[mp+'_trans']=j[:,0].tolist();row[mp+'_global_orient']=h[side+'_global_orient'].tolist();row[mp+'_finger_pose']=h[side+'_finger_pose'].tolist()
  row[mp+'_joints']=j.tolist();row[mp+'_joints_wxyz']=q.tolist();row[mp+'_valid']=h[side+'_valid'].tolist();row[mp+'_interpolated']=h[side+'_interpolated'].tolist();row[mp+'_fitting_err']=np.zeros(N).tolist()
  # Drop stale WiLoR-derived proximity labels. DexAgent computes geometric/physical contacts.
  for k in list(row):
   if k.startswith(mp) and ('contact_' in k or 'tips_distance' in k):row[k]=np.zeros_like(np.asarray(row[k])).tolist()
  data[side+'_qpos']=np.c_[pose[:,:3],pose[:,[6,3,4,5]],pose[:,7:]];data[side+'_raw_qpos_xyzw']=raw
  data[side+'_joint_names']=np.array(row[side+'_robot_finger_joint_names']);data[side+'_valid']=h[side+'_valid']
  report['sides'][side]={'model_xml':str(SHARPA_WAVE_XMLS_DIR/f'{side}_sharpawave.xml'),'model_xml_sha256':sha(SHARPA_WAVE_XMLS_DIR/f'{side}_sharpawave.xml'),'target_error_mean_m':float(np.mean(errors)),'target_error_p95_m':float(np.percentile(errors,95)),'valid_frames':int(h[side+'_valid'].sum()),'frames':N}
  handj.append(j);handr.append(R.from_quat(q.reshape(-1,4)[:,[1,2,3,0]]).as_matrix().reshape(N,21,3,3))
 row['raw_motion_file']=str(out/'hawor_tracks.json');row['mano_flat_hand_mean']=True;row['mano_center_idx']=None;row['mano_to_robot_scale']=None
 row['object_mesh_paths']=[str(out/'assets'/Path(x).name) for x in row['object_mesh_paths']];row['object_urdf_paths']=[str(out/'assets'/Path(x).name) for x in row['object_urdf_paths']]
 # Use the original schema; every object-pose scalar must stay exactly unchanged.
 pq.write_table(pa.Table.from_pylist([row],schema=table.schema),out/'reference.parquet')
 read=pq.ParquetFile(out/'reference.parquet').read().to_pylist()[0];original=table.to_pylist()[0]
 for k in ['object_body_names','safe_object_body_names','object_body_position','object_body_wxyz','object_root_position','object_root_axis_angle','object_articulation']:
  assert read[k]==original[k],k
 np.savez_compressed(out/'sharpa_ik.npz',**data)
 np.savez_compressed(out/'hand_targets.npz',hand_joints=np.stack(handj,1),hand_orientations=np.stack(handr,1),hand_valid=np.stack([h[s+'_valid'] for s in ['left','right']],1),hand_interpolated=np.stack([h[s+'_interpolated'] for s in ['left','right']],1),fps=20.)
 np.savez_compressed(out/'hand_validity.npz',valid=np.stack([h[s+'_valid'] for s in ['left','right']],1),interpolated=np.stack([h[s+'_interpolated'] for s in ['left','right']],1))
 report['decode']=load(out/'hawor_decode_report.json');report['reference_sha256']=sha(out/'reference.parquet');report['smoothing']={'mano_window_frames':9,'robot_window_frames':11,'polyorder':2}
 dump(out/'hand_report.json',report)
 config=load(src/'config.json');config['hand_source']='HaWoR camera-space depth-aligned tracks; see hand_report.json';config['robot']='Exact Sharpa Wave XML, floating hands, Mink IK';config['frame_name_note']='All sites and body poses recomputed by exact-model FK';dump(out/'config.json',config)
 dump(out/'validation.json',{'frames':N,'finite':bool(all(np.isfinite(x).all() for k,x in data.items() if x.dtype.kind in 'fc')),'objects_preserved':True,'hand_source':'HaWoR','joint_limits_ok':True,'physics_validated':False})
 (out/'USAGE.txt').write_text('HaWoR hands + supplied final_recon_ours object reconstruction.\nInput WiLoR-only hands/videos are preserved under archive_wilor/.\nreference.parquet is exact Sharpa Wave FK/IK, 20 Hz, table-aligned z up.\nMissing HaWoR hand frames are world-interpolated/endpoint held and flagged invalid.\nNo physical grasp success is claimed until DexAgent rollout validation.\n')
 dump(out/'hawor_complete.json',{'episode':e,'reference':str(out/'reference.parquet'),'reference_sha256':sha(out/'reference.parquet'),'hand_source':'HaWoR','objects_preserved':True,'physics_validated':False})
 print('completed HaWoR',e,report['sides'],flush=True)

def docker_phase(phase,episodes):
 import shlex
 image='v2d_task_library_loader:latest' if phase=='decode' else 'robotic-grounding:mink-1.3.0'
 entry='python' if phase=='decode' else '/workspace/isaaclab/_isaac_sim/python.sh'
 cmd=['docker','run','--rm','--user',f'{os.getuid()}:{os.getgid()}','--group-add','1234','-e',f'TASK_UID={os.getuid()}','-e',f'TASK_GID={os.getgid()}','-e','HOME=/tmp','-e','CUDA_VISIBLE_DEVICES=','-e','NVIDIA_VISIBLE_DEVICES=void','-e',f'HAWOR_OUTPUT_ROOT={OUTROOT}','-e','PYTHONUNBUFFERED=1','-e','OMP_NUM_THREADS=2','--entrypoint',entry,'-v','/data1:/data1']
 if phase=='ik':cmd+=['-v',str(ROOT/'robotic_grounding')+':/workspace/video_to_data/robotic_grounding','-w','/workspace/video_to_data/robotic_grounding']
 cmd += [image,str(Path(__file__).resolve()),'--phase',phase,'--episodes',*[str(x) for x in episodes]]
 subprocess.run(['sg','docker','-c',shlex.join(cmd)],check=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['all','prepare','decode','ik'],default='all');p.add_argument('--episodes',type=int,nargs='+',default=EPISODES);a=p.parse_args()
 if a.phase in ['all','prepare']:
  shim=OLD_MINK.read_text().split('\ninstances=[]')[0]+'\n'
  (BASE/'code/mink_sharpa_exact.py').write_text(shim)
  for e in a.episodes:prepare(e)
 if a.phase=='all':
  docker_phase('decode',a.episodes);docker_phase('ik',a.episodes)
 elif a.phase=='decode':
  for e in a.episodes:decode(e)
 elif a.phase=='ik':
  for e in a.episodes:ik(e)
if __name__=='__main__':
 try:main()
 finally:
  if os.geteuid()==0 and 'TASK_UID' in os.environ:
   for p in (BASE/'hawor_recon').rglob('*'):
    if p.stat().st_uid==0:os.chown(p,int(os.environ['TASK_UID']),int(os.environ['TASK_GID']))
