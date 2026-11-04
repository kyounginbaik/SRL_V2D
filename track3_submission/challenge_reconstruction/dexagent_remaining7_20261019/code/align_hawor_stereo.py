#!/usr/bin/env python3
"""Fit only HaWoR metric scale and camera translation to calibrated stereo depth.

No WiLoR joint/pose arrays are read. Original HaWoR outputs stay in hawor_recon.
All 45 local finger rotation parameters and 21 world joint orientations stay fixed.
"""
from pathlib import Path
import argparse,json,hashlib,shutil,os,subprocess,concurrent.futures
import numpy as np,cv2,trimesh
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree
B=Path(__file__).resolve().parents[1]
ROOT=Path('/data1/SRL_V2D')
TARGET=Path(os.environ.get('HAWOR_STEREO_OUTPUT_ROOT',str(B/'hawor_stereo_recon_v2')))

def load(p):return json.loads(Path(p).read_text())
def save(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def xyz_cam(world,C):return np.einsum('nji,nkj->nki',C[:,:3,:3],world-C[:,None,:3,3])
def xyz_world(cam,C):return np.einsum('nij,nkj->nki',C[:,:3,:3],cam)+C[:,None,:3,3]
def mesh_distance(joints,O,meshes):
 distances=[]
 for i,mesh in enumerate(meshes):
  local=np.einsum('nji,nkj->nki',O[:,i,:3,:3],joints-O[:,i,None,:3,3])
  dist=cKDTree(mesh.vertices).query(local.reshape(-1,3))[0].reshape(len(joints),-1)
  distances.append(dist)
 return np.stack(distances,axis=1)

def align(e):
 src=B/f'hawor_recon/episode_{e:06d}';out=TARGET/f'episode_{e:06d}'
 if out.exists():
  if (out/'stereo_fit.json').exists():return
  raise RuntimeError(f'Inspect partial output {out}')
 shutil.copytree(src,out)
 for name in ['hawor_complete.json','validation.json']:(out/name).unlink(missing_ok=True)
 archive=out/'archive_before_stereo';archive.mkdir()
 for name in ['reference.parquet','sharpa_ik.npz','hand_targets.npz','hand_validity.npz','hawor_decoded.npz','hand_report.json','hawor_keypoint_review.jpg','hawor_keypoint_review.json']:
  q=out/name
  if q.exists():shutil.move(q,archive/name)
 h=dict(np.load(src/'hawor_decoded.npz'));C=np.load(src/'camera_motion.npz')['world_from_rgb'].astype(float);N=len(C)
 meta=load(src/'hawor_tracks.json');origin=Path(meta['source']);ki=load(origin/'intrinsics_stable.json')
 # Read the exact K used by the cached depth projector, not HaWoR's
 # averaged/stabilized intrinsics (which differ by several pixels).
 import re
 projection_script=ROOT/f'challenge_reconstruction/eval_stereo_current_20261010/episode_{e:06d}/project_depth.py'
 projection_origin=Path(re.search(r"old=Path\('([^']+)'\)",projection_script.read_text()).group(1))
 depth_intrinsics=projection_origin/'anycalib/undistorted_intrinsics.json'
 kd_full=load(depth_intrinsics)
 K=np.array([[kd_full['fx'],0,kd_full['cx']],[0,kd_full['fy'],kd_full['cy']],[0,0,1.]])
 depthroot=ROOT/f'challenge_reconstruction/eval_stereo_current_20261010/episode_{e:06d}/depth_rgb'
 maskmap={}
 mask_tracks=load(origin/'hand_tracks.json')['tracks']
 for side in ['left','right']:
  primary=origin/'masks'/str(meta['source_metadata'][side+'_id'])
  if primary.is_dir():maskmap[side]=primary
  else:
   candidates=[origin/'masks'/str(t['object_id']) for t in mask_tracks if t.get('role')=='hand' and bool(t['is_right'])==(side=='right')]
   candidates=[q for q in candidates if q.is_dir()]
   if len(candidates)==1:maskmap[side]=candidates[0]
 all_depth=[]
 for frame in range(N):
  encoded=cv2.imread(str(depthroot/f'{frame:06d}.png'),-1);assert encoded is not None
  depth=np.zeros(encoded.shape,np.float64);valid=(encoded>0)&(encoded<65535);depth[valid]=65535./encoded[valid]-1.
  depth[(depth<.08)|(depth>2.)]=0;all_depth.append(depth)
 H,W=all_depth[0].shape;Kd=K.copy();Kd[0]*=W/kd_full['width'];Kd[1]*=H/kd_full['height']
 O=np.load(src/'object_poses.npz');meshes=[trimesh.load(src/f'assets/{name}.obj',force='mesh',process=False) for name in O['names']]
 report={'episode':e,'method':'one metric scale per hand + smooth framewise camera-ray translation from cached calibrated stereo depth','No_WiLoR_joint_or_pose_inputs':True,'finger_articulation_modified':False,'object_poses_modified':False,'depth_root':str(depthroot),'depth_encoding':'valid raw uint16 1..65534 => 65535/value - 1 meters; 0 and65535invalid','depth_grid_shape':[H,W],'depth_K':Kd.tolist(),'source_image_shape':[ki['height'],ki['width']],'source_K':K.tolist(),'depth_projection_intrinsics':str(depth_intrinsics),'depth_projection_intrinsics_sha256':sha(depth_intrinsics),'depth_projection_script_sha256':sha(projection_script),'hawor_stable_K':ki,'depth_grid_K_is_exact':True,'sides':{}}
 for side in ['left','right']:
  vertices=xyz_cam(h[side+'_vertices_world'],C);joints=xyz_cam(h[side+'_joints_world'],C)
  measurements=[];counts=np.zeros(N,int);masks=[]
  for f in range(N):
   pts=vertices[f];uv=pts@Kd.T;uv=uv[:,:2]/uv[:,2:3];xy=np.rint(uv).astype(int);valid=(pts[:,2]>.05)&(xy[:,0]>=0)&(xy[:,0]<W)&(xy[:,1]>=0)&(xy[:,1]<H)
   ix=np.flatnonzero(valid);sample=all_depth[f][xy[ix,1],xy[ix,0]]
   keep=sample>.08
   if side in maskmap:
    mask=cv2.imread(str(maskmap[side]/f'{f:06d}.png'),0)
    if mask is not None:
     mask=cv2.resize(mask,(W,H),interpolation=cv2.INTER_NEAREST)>0
     # Only retain a hand-segmentation mask when overlap exists. Fully occluded
     # frames remain unsupported instead of using background depth as evidence.
     keep &= mask[xy[ix,1],xy[ix,0]]
   ix=ix[keep];sample=sample[keep]
   # Trim far background and incorrect stereo matches using ratio dispersion.
   ratio=sample/pts[ix,2]
   if len(ratio):
    med=np.median(ratio);mad=np.median(np.abs(ratio-med));good=np.abs(ratio-med)<max(.06,3*mad)
    ix=ix[good];sample=sample[good]
   if not h[side+'_valid'][f]:ix=ix[:0];sample=sample[:0]
   counts[f]=len(ix);measurements.append((ix,sample))
  good_frames=np.flatnonzero(counts>=40)
  if len(good_frames)<10:raise RuntimeError(f'Insufficient stereo support {e} {side} {len(good_frames)}')
  ratios=np.array([np.median(d/vertices[f,ix,2]) if len(ix)>=40 else np.nan for f,(ix,d) in enumerate(measurements)])
  scale=float(np.nanmedian(ratios));assert .3<scale<1.5,(e,side,scale)
  # Scaling around the camera origin preserves projected geometry before the
  # small per-frame residual translation. This corrects the old monocular scale.
  vnew=vertices*scale;jnew=joints*scale;residual=np.full(N,np.nan)
  for f,(ix,d) in enumerate(measurements):
   if len(ix)>=40:residual[f]=np.median(d-vnew[f,ix,2])
  good=np.flatnonzero(np.isfinite(residual));dz=np.interp(np.arange(N),good,residual[good]);dz=savgol_filter(dz,min(11,N//2*2-1),2)
  centroid=vnew.mean(axis=1);ray=centroid/centroid[:,2:3];translation=ray*dz[:,None]
  vnew+=translation[:,None,:];jnew+=translation[:,None,:]
  after_res=[];before_res=[]
  for f,(ix,d) in enumerate(measurements):
   if len(ix)>=40:after_res.extend((d-vnew[f,ix,2]).tolist());before_res.extend((d-vertices[f,ix,2]).tolist())
  new_worldj=xyz_world(jnew,C);new_worldv=xyz_world(vnew,C)
  # Gap interpolation is redone after metric correction, in world coordinates.
  avail=h[side+'_available'];ix=np.flatnonzero(avail)
  if (~avail).any():
   for values in [new_worldj,new_worldv]:
    flat=values.reshape(N,-1)
    for col in range(flat.shape[1]):flat[:,col]=np.interp(np.arange(N),ix,flat[ix,col])
  beforedist=mesh_distance(h[side+'_joints_world'],O['world_from_object'],meshes);afterdist=mesh_distance(new_worldj,O['world_from_object'],meshes)
  h[side+'_vertices_world']=new_worldv;h[side+'_joints_world']=new_worldj;h[side+'_hand_scale']*=scale
  side_report={'global_camera_scale':scale,'effective_hand_scale_median':float(np.median(h[side+'_hand_scale'])),'stereo_supported_frames':len(good),'stereo_samples_per_frame':counts.tolist(),'hand_mask_path':str(maskmap.get(side,'none; projected HaWoR vertices fallback')),'before_surface_depth_abs_median_m':float(np.median(np.abs(before_res))),'after_surface_depth_abs_median_m':float(np.median(np.abs(after_res))),'camera_translation_delta_m_p50':float(np.median(np.linalg.norm(jnew[:,0]-joints[:,0],axis=1))),'framewise_residual_dz_range_m':[float(dz.min()),float(dz.max())],'allhandnearestobject_surface_distance_before_median_m':float(np.median(beforedist.min(axis=(1,2)))),'allhandnearestobject_surface_distance_after_median_m':float(np.median(afterdist.min(axis=(1,2)))),'wrist_world_z_before_range_m':[float(h[side+'_joints_world'][:,0,2].min()),float(h[side+'_joints_world'][:,0,2].max())],'observed_finger_parameters_sha256':hashlib.sha256(h[side+'_finger_pose'].tobytes()).hexdigest()}
  # Every local finger angle and global hand orientation stays byte-identical.
  orig=np.load(src/'hawor_decoded.npz')
  for key in ['finger_pose','joints_wxyz','global_orient','betas']:assert np.array_equal(h[side+'_'+key],orig[side+'_'+key]),(side,key)
  side_report['wrist_world_z_before_range_m']=[float(orig[side+'_joints_world'][:,0,2].min()),float(orig[side+'_joints_world'][:,0,2].max())]
  side_report['wrist_world_z_after_range_m']=[float(new_worldj[:,0,2].min()),float(new_worldj[:,0,2].max())]
  report['sides'][side]=side_report
  np.savez_compressed(out/f'stereo_fit_{side}.npz',scale=scale,frame_dz=dz,counts=counts,frame_ratios=ratios,nearest_object_distances_before=beforedist,nearest_object_distances_after=afterdist)
 np.savez_compressed(out/'hawor_decoded.npz',**h)
 prior=load(src/'hawor_decode_report.json');prior['stereo_metric_alignment']=report;save(out/'hawor_decode_report.json',prior);save(out/'stereo_fit.json',report)
 print('STEREO FIT',e,{s:{k:v for k,v in d.items() if k in ['global_camera_scale','effective_hand_scale_median','after_surface_depth_abs_median_m','allhandnearestobject_surface_distance_before_median_m','allhandnearestobject_surface_distance_after_median_m']} for s,d in report['sides'].items()},flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('episodes',nargs='+',type=int);p.add_argument('--ik',action='store_true');a=p.parse_args()
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
  for f in concurrent.futures.as_completed([ex.submit(align,e) for e in a.episodes]):f.result()
 if a.ik:
  os.environ['HAWOR_OUTPUT_ROOT']=str(TARGET)
  import importlib.util
  spec=importlib.util.spec_from_file_location('prep',B/'code/prepare_hawor_hands.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
   for f in concurrent.futures.as_completed([ex.submit(m.docker_phase,'ik',[e]) for e in a.episodes]):f.result()
if __name__=='__main__':main()
