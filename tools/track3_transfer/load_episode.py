"""Build a shared-frame, multi-object ManoSharpaData reference from video bundles."""
import sys,json
from pathlib import Path
import numpy as np
import trimesh
from v2d.task_library_loader.lib.ego_recon_loader import EgoReconDatasetLoader,parse_args
from v2d.task_library_loader.lib.dataset_loader_base import load_meshes_to_device,poses_to_root_position_and_axis_angle
BASE=Path(__file__).resolve().parent.parent
RECON=BASE.parent/'eval_hawor_latest_20261008'
episode=sys.argv[1];rows=[r for r in json.loads((RECON/'manifest.json').read_text()) if r['episode']==episode]
assert rows
rows.sort(key=lambda r:r['slot']);roots=[Path(r['output'])/'result_slam_gravity_aligned' for r in rows]
bundles=[dict(np.load(p/'result.npz')) for p in roots];camera=bundles[0]['camera_to_world_transform']
assert all(len(b['object_to_camera_transform'])==len(camera) for b in bundles)
run=BASE/episode;assets=run/'assets';assets.mkdir(parents=True,exist_ok=True)
class MultiLoader(EgoReconDatasetLoader):
 def list_sequences(self,args):
  seq=super().list_sequences(args);assert len(seq)==1
  seq[0].object_body_names=[r['object'] for r in rows]
  return seq
 def get_fps(self):return 20.0
 def _scene_align(self,seq):
  if seq.sequence_id not in self._align_cache:
   verts=[]
   for root,b in zip(roots,bundles):
    m=trimesh.load(root/'mesh.obj',force='mesh');T=camera[0]@b['object_to_camera_transform'][0]
    verts.append(m.vertices@T[:3,:3].T+T[:3,3])
   v=np.concatenate(verts);t=np.array([-v[:,0].mean(),-v[:,1].mean(),-v[:,2].min()])
   self._align_cache[seq.sequence_id]=(np.eye(3),t)
   (run/'placement.json').write_text(json.dumps({'translation':t.tolist(),'rotation':np.eye(3).tolist(),'objects':[r['object'] for r in rows],'fps':20,'camera_source':str(roots[0]),'ground_estimate':'lowest reconstructed object vertex at first frame'}))
  return self._align_cache[seq.sequence_id]
 def load_object_data(self,seq):
  R,t=self._scene_align(seq);result={}
  for r,b in zip(rows,bundles):
   poses=(camera@b['object_to_camera_transform']).astype(np.float64);poses[:,:3,3]+=t
   pos,aa=poses_to_root_position_and_axis_angle(poses);result[r['object']]=(poses,pos,aa,None)
  return result
 def load_object_meshes(self,seq,device):
  return load_meshes_to_device({r['object']:str(p/'mesh.obj') for r,p in zip(rows,roots)},device)
 def get_object_mesh_paths(self,seq):
  sys.path.insert(0,'/data1/SRL_V2D/tools')
  from track3_merge_objects import _write_assets
  return [str(_write_assets(root/'mesh.obj',assets,r['object'],1.)[0]) for r,root in zip(rows,roots)]
 def get_object_urdf_paths(self,seq):return [str(assets/(r['object']+'.urdf')) for r in rows]
sys.argv=[sys.argv[0],'--dataset_root',str(roots[0].parent),'--result_subpath','result_slam_gravity_aligned','--object_name',rows[0]['object'],'--sequence_name',episode,'--mesh_dir',str(assets),'--object_model_root',str(assets),'--output_dir',str(run/'loaded'),'--mano_model_dir','/data1/SRL_V2D/reconstruction/data/weights/hamer/_DATA/data','--device','cuda:0','--no_ground_align','--save']
MultiLoader().run(parse_args())
(run/'load_complete.json').write_text(json.dumps({'objects':[r['object'] for r in rows],'frames':len(camera),'fps':20}))
