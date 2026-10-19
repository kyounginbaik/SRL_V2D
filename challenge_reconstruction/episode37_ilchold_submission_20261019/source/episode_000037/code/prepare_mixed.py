"""Materialize the reviewed mixed IK candidate without reusing stale contacts."""
from pathlib import Path
import json,hashlib,shutil,xml.etree.ElementTree as ET
import numpy as np,pyarrow as pa,pyarrow.parquet as pq,trimesh,mujoco
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R,Slerp
B=Path(__file__).resolve().parents[1]
P=B.parent/'final_recon_candidates/episode_000001_coworker_hawor_ik'
A=Path('/data1/SRL_V2D/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/xmls/sharpawave')
F=B/'frozen/episode_000001';assets=F/'assets';assets.mkdir(parents=True,exist_ok=True)
D=np.load(P/'comparison_inputs.npz');Q=np.load(P/'sharpa_ik.npz');O=np.load(P/'mixed_object_poses.npz')['object_poses'];meta=json.loads((P/'mixed_inputs.json').read_text());T=len(O)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
r={'fps':20.,'raw_motion_file':str(P/'mixed_object_poses.npz'),'source_dataset':'mixed_reconstruction','object_name':'white_pot','object_body_names':['white_pot','white_pot_lid'],'safe_object_body_names':['white_pot','white_pot_lid'],'object_body_position':O[:,:,:3,3].tolist(),'object_body_wxyz':R.from_matrix(O[:,:,:3,:3].reshape(-1,3,3)).as_quat(scalar_first=True).reshape(T,2,4).tolist(),'object_articulation':np.zeros(T).tolist()}
meshes=[];samples=[];trees=[];normals=[]
for name,source in zip(r['object_body_names'],[meta['pot']['mesh'],meta['lid']['mesh']]):
 target=assets/(name+'.obj');shutil.copy2(source,target);m=trimesh.load(target,force='mesh');meshes.append(m)
 pts,face=trimesh.sample.sample_surface(m,200000,seed=42);samples.append(pts);trees.append(cKDTree(pts));normals.append(m.face_normals[face])
r['object_mesh_paths']=[str(assets/(n+'.obj')) for n in r['object_body_names']];r['object_urdf_paths']=[str(assets/(n+'.urdf')) for n in r['object_body_names']];r['object_mesh_radius']=[float(np.linalg.norm(m.vertices-m.centroid,axis=1).max()) for m in meshes]
report={'source_video':str(P/'mixed_ik_ego.mp4'),'frames':T,'fps':20,'physics_validated':False,'hands':{},'contact_method':'Nearest dense sampled reconstructed surface to HaWoR 21 joints, <=20mm; no motion-only contact inference, labels disabled for missing hand frames. Heuristic geometric estimates, not observed physical forces.','contact_threshold_m':.02,'objects_unchanged_from_reviewed_mix':True,'nominal_mass_kg_per_object':.3,'mass_note':'Inherited nominal 0.3kg; not measured. Inertia approximated using uniform surface samples.'}
filled={}
for h,side in enumerate(['left','right']):
 q=Q[side+'_qpos'].copy();valid=Q[side+'_valid'];known=np.flatnonzero(valid);missing=np.flatnonzero(~valid)
 assert known[0]==0 and known[-1]==T-1
 for j in [0,1,2,*range(7,q.shape[1])]:q[missing,j]=np.interp(missing,known,q[known,j])
 if len(missing):q[missing,3:7]=Slerp(known,R.from_quat(q[known,3:7],scalar_first=True))(missing).as_quat(scalar_first=True)
 assert np.isfinite(q).all();assert np.array_equal(q[valid],Q[side+'_qpos'][valid]);filled[side+'_qpos']=q;filled[side+'_original_valid']=valid
 root=ET.parse(A/f'{side}_sharpawave.xml').getroot()
 for parent in root.iter():
  for child in list(parent):
   if child.tag in ('geom','asset','actuator','sensor','contact','equality','keyframe'):parent.remove(child)
 ET.SubElement(root.find('worldbody/body'),'freejoint',name='floating_root');model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'));data=mujoco.MjData(model)
 specs=[('body',i,model.body(i).name) for i in range(1,model.nbody)]+[('site',i,model.site(i).name) for i in range(model.nsite)]
 specs=[s for s in specs if s[2]];assert len({s[2] for s in specs})==len(specs)
 frames=np.empty((T,len(specs),7))
 for f in range(T):
  data.qpos[:]=q[f];mujoco.mj_forward(model,data)
  for i,(typ,idx,name) in enumerate(specs):
   frames[f,i,:3]=data.xpos[idx] if typ=='body' else data.site_xpos[idx]
   frames[f,i,3:]=data.xquat[idx] if typ=='body' else R.from_matrix(data.site_xmat[idx].reshape(3,3)).as_quat(scalar_first=True)
 r[f'robot_{side}_wrist_position']=q[:,:3].tolist();r[f'robot_{side}_wrist_wxyz']=q[:,3:7].tolist();r[f'robot_{side}_finger_joints']=q[:,7:].tolist();r[f'{side}_robot_finger_joint_names']=Q[side+'_joint_names'].tolist();r[f'{side}_robot_frame_names']=[s[2] for s in specs];r[f'robot_{side}_frames']=frames.tolist()
 cp=np.zeros((T,21,3));cn=cp.copy();ids=np.zeros((T,21),int);distance=np.full((T,21),np.inf)
 for f in known:
  hands=D['hand_joints'][f,h]
  for o in range(2):
   rot=O[f,o,:3,:3];trans=O[f,o,:3,3];local=(hands-trans)@rot;dist,idx=trees[o].query(local);take=(dist<distance[f])&(dist<=.02)
   cp[f,take]=samples[o][idx[take]]@rot.T+trans;cn[f,take]=normals[o][idx[take]]@rot.T;ids[f,take]=o+1;distance[f,take]=dist[take]
 for name,val in [('positions',cp),('normals',cn),('part_ids',ids)]:r[f'mano_{side}_object_contact_{name}']=val.tolist()
 report['hands'][side]={'missing_interpolated_frames':missing.tolist(),'missing_completion':'linear wrist/fingers and quaternion SLERP between observed endpoints; contacts off','observed_ik_preserved_exactly':True,'active_contact_frames_per_object':{n:int(np.any(ids==o+1,axis=1).sum()) for o,n in enumerate(r['object_body_names'])},'max_wrist_step_m':float(np.linalg.norm(np.diff(q[:,:3],axis=0),axis=1).max())}
np.savez_compressed(F/'completed_ik.npz',**filled)
pq.write_table(pa.Table.from_pylist([r]),F/'reference.parquet');shutil.copy2(F/'reference.parquet',F/'approved_reference.parquet')
report['reference_sha256']=digest(F/'reference.parquet');report['source_hashes']={str(p):digest(p) for p in [P/'mixed_object_poses.npz',P/'sharpa_ik.npz',P/'comparison_inputs.npz']}
(F/'preparation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['hands'],indent=2),flush=True)
# Nominal inertial approximation in each mesh's own local coordinate system.
for o,name in enumerate(r['object_body_names']):
 pts=samples[o];com=pts.mean(0);x=pts-com;inertia=.3*(np.eye(3)*np.mean(np.sum(x*x,axis=1))-x.T@x/len(x))
 robot=ET.Element('robot',name=name);link=ET.SubElement(robot,'link',name='object');ii=ET.SubElement(link,'inertial');ET.SubElement(ii,'origin',xyz=' '.join(map(str,com)),rpy='0 0 0');ET.SubElement(ii,'mass',value='.3');ET.SubElement(ii,'inertia',**{k:str(inertia[i,j]) for k,i,j in [('ixx',0,0),('ixy',0,1),('ixz',0,2),('iyy',1,1),('iyz',1,2),('izz',2,2)]});v=ET.SubElement(link,'visual');ET.SubElement(ET.SubElement(v,'geometry'),'mesh',filename=name+'.obj')
 if o==1:
  scale=float(np.load(B.parent/'final_recon_candidates/episode_000001_object_pose_audit/candidate_object_poses.npz')['mesh_scale'])
  source=B.parent/'final_recon/episode_000001/assets/white_pot_lid_convex';out=assets/(name+'_convex');out.mkdir(exist_ok=True)
  for p in sorted(source.glob('*.obj')):
   m=trimesh.load(p,force='mesh');m.apply_scale(scale);m.export(out/p.name)
 else:
  import coacd
  coacd.set_log_level('warn');m=meshes[o]
  if len(m.faces)>30000:m=m.simplify_quadric_decimation(face_count=30000)
  out=assets/(name+'_convex');out.mkdir(exist_ok=True)
  parts=coacd.run_coacd(coacd.Mesh(np.asarray(m.vertices),np.asarray(m.faces)),threshold=.05,seed=42)
  for j,(v,f) in enumerate(parts):trimesh.Trimesh(v,f,process=False).export(out/f'part_{j:03d}.obj')
 for p in sorted(out.glob('*.obj')):ET.SubElement(ET.SubElement(ET.SubElement(link,'collision'),'geometry'),'mesh',filename=str(p.relative_to(assets)))
 assert len(list(out.glob('*.obj')))>1
 ET.indent(robot);ET.ElementTree(robot).write(assets/(name+'.urdf'),encoding='utf-8',xml_declaration=True)
 print(name,'CoACD pieces',len(list(out.glob('*.obj'))),flush=True)
manifest=[{'episode':'episode_000001','runtime_sha256':digest(F/'reference.parquet'),'approved_sha256':digest(F/'approved_reference.parquet'),'asset_sha256':{str(p.relative_to(F)):digest(p) for p in sorted(assets.rglob('*')) if p.is_file()},'frames':T,'fps':20,'objects':r['object_body_names']}]
(B/'frozen_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print('PREPARED',flush=True)
