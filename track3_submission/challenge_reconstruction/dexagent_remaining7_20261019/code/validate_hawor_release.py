#!/usr/bin/env python3
from pathlib import Path
import json,hashlib,numpy as np,pyarrow.parquet as pq,mujoco
B=Path(__file__).resolve().parents[1];ROOT=Path('/data1/SRL_V2D');OUT=B/'hawor_stereo_recon_v2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
models={s:mujoco.MjModel.from_xml_path(str(ROOT/f'robotic_grounding/source/robotic_grounding/robotic_grounding/assets/xmls/sharpawave/{s}_sharpawave.xml')) for s in ['left','right']}
report={'source_root':str(OUT),'hand_source':'HaWoR + calibrated stereo metric alignment','episodes':{},'physics_success_certified':False,'notes':['These checks certify input kinematics and provenance, not physical rollout success.','Nearest-hand-joint mesh distances use source mesh vertices and are not contact force measurements.']}
for e in [14,15,16,20,25,28,30]:
 p=OUT/f'episode_{e:06d}';src=B/f'source/final_recon_ours/episode_{e:06d}';original=pq.ParquetFile(src/'reference.parquet').read().to_pylist()[0];row=pq.ParquetFile(p/'reference.parquet').read().to_pylist()[0];assert (p/'hawor_complete.json').exists()
 record={'reference_sha256':sha(p/'reference.parquet'),'object_poses_sha256':sha(p/'object_poses.npz'),'camera_sha256':sha(p/'camera_motion.npz'),'stereo_fit_sha256':sha(p/'stereo_fit.json'),'source_track_records_sha256':sha(p/'hawor_tracks.json'),'preserved_meshes':{},'sides':{}}
 assert sha(src/'object_poses.npz')==record['object_poses_sha256'];assert sha(src/'camera_motion.npz')==record['camera_sha256']
 for key in ['object_body_position','object_body_wxyz','object_root_position','object_root_axis_angle','object_articulation']:assert row[key]==original[key],(e,key)
 for mesh in (src/'assets').glob('*.obj'):
  assert sha(mesh)==sha(p/'assets'/mesh.name),(e,mesh)
  record['preserved_meshes'][mesh.name]=sha(mesh)
 old=np.load(B/f'hawor_recon/episode_{e:06d}/hawor_decoded.npz');new=np.load(p/'hawor_decoded.npz')
 for s in ['left','right']:
  model=models[s];qs=np.array(row['robot_'+s+'_finger_joints']);names=row[s+'_robot_finger_joint_names'];assert np.isfinite(qs).all()
  violation=0.
  for i,name in enumerate(names):
   jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name);assert jid>=0
   if model.jnt_limited[jid]:violation=max(violation,float(np.maximum(model.jnt_range[jid,0]-qs[:,i],qs[:,i]-model.jnt_range[jid,1]).max()))
  assert violation<1e-7,(e,s,violation)
  quat=np.array(row['robot_'+s+'_wrist_wxyz']);assert np.max(np.abs(np.linalg.norm(quat,axis=1)-1))<1e-7
  for key in ['finger_pose','joints_wxyz','global_orient','betas']:assert np.array_equal(old[s+'_'+key],new[s+'_'+key]),(e,s,key)
  fit=json.loads((p/'stereo_fit.json').read_text())['sides'][s]
  record['sides'][s]={'joint_limit_violation_max_rad':violation,'HaWoR_finger_articulation_byte_identical':True,'frame_count':len(qs),'global_camera_scale':fit['global_camera_scale'],'effective_hand_scale_median':fit['effective_hand_scale_median'],'depth_residual_median_m':fit['after_surface_depth_abs_median_m'],'initial_robot_wrist_position_m':row['robot_'+s+'_wrist_position'][0],'initial_robot_hand_lowest_frame_z_m':float(np.array(row['robot_'+s+'_frames'])[0,:,2].min())}
 report['episodes'][str(e)]=record
(OUT/'release_ready.json').write_text(json.dumps(report,indent=2)+'\n');(B/'review/hawor_final_validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({e:r['reference_sha256'] for e,r in report['episodes'].items()},indent=2))
