"""Bind exported body-frame meshes to an actually simulated repaired scene.

This makes a new adapter directory. Immutable source trajectories, evaluation
anchors and earlier adapters remain byte-for-byte unchanged.
"""
from pathlib import Path
import argparse,json,shutil,xml.etree.ElementTree as ET
import numpy as np,mujoco,trimesh
from scipy.spatial import cKDTree
from learned_feedback_policy import sha
B=Path(__file__).resolve().parents[1]
def prepare(episode,run,output):
 source=B/f'episodes/episode_{episode:06d}';out=Path(output).resolve();run=Path(run).resolve()
 if (out/'input_manifest.json').exists():raise FileExistsError(out)
 out.mkdir(parents=True,exist_ok=True);r=json.loads((run/'result.json').read_text());scene=Path(r['execution']['scene']);manifest=json.loads((source/'input_manifest.json').read_text());tree=ET.parse(scene);meshdefs={x.get('name'):x for x in tree.findall('.//asset/mesh')};model=mujoco.MjModel.from_xml_path(str(scene));data=mujoco.MjData(model);mujoco.mj_forward(model,data);records=[]
 for name in ['tracking_reference.npz','reference_trajectory.npz','camera_trajectory.npz','metric_geometry.json','calibrated_frame_validation.json']:
  if (source/name).exists():shutil.copy2(source/name,out/name)
 for obj in manifest['objects']:
  name=obj['id'];body=tree.find(f'.//body[@name="{name}"]');bid=model.body(name).id;original=obj['meshes'];actual=[];visual_count=0
  for geom in body.findall('geom'):
   if geom.get('type')!='mesh':continue
   md=meshdefs[geom.get('mesh')]
   # Current adapters express geometry directly in the object's frame. Fail
   # closed rather than export unbaked transforms as if they were absent.
   if any(k in geom.attrib for k in ['pos','quat','euler','axisangle','xyaxes','zaxis']):raise ValueError('Geometry transform must be explicitly baked before export')
   if md.get('scale','1 1 1')!='1 1 1':raise ValueError('Unbaked mesh scale')
   p=Path(md.get('file'));p=p if p.is_absolute() else scene.parent/p
   kind='visual' if geom.get('contype')=='0' and geom.get('conaffinity')=='0' else 'collision'
   actual.append(dict(source=str(p.resolve()),sha256=sha(p),kind=kind,scene_geom=geom.get('name')))
   if kind=='visual':
    visual_count+=1;loaded=trimesh.load(p,process=False);verts=np.asarray(loaded.vertices);gid=model.geom(geom.get('name')).id;mid=int(model.geom_dataid[gid]);v=model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid]+model.mesh_vertnum[mid]];world=v@data.geom_xmat[gid].reshape(3,3).T+data.geom_xpos[gid];bodyverts=(world-data.xpos[bid])@data.xmat[bid].reshape(3,3)
    error=max(cKDTree(verts).query(bodyverts)[0].max(),cKDTree(bodyverts).query(verts)[0].max());assert error<1e-6,(name,error)
    old=[x for x in original if x['kind']=='visual'][0];records.append(dict(object=name,actual_visual_mesh=str(p),actual_visual_sha256=sha(p),original_visual=old,changed=sha(p)!=old['sha256'],body_frame_compiled_mesh_max_bidirectional_distance_m=float(error),vertices=len(verts),faces=len(loaded.faces),xml_scale_baked=True,xml_visual_geom_transform_identity=True))
  assert visual_count==1;obj['original_mesh_records']=original;obj['meshes']=actual;obj['simulated_visual_geometry_verified']=True
 manifest.update(original_adapter_dir=str(source),original_adapter_manifest_sha256=sha(source/'input_manifest.json'),physical_scene=str(scene),physical_scene_sha256=sha(scene),repaired_mesh_export_binding=records,source_object_targets_and_original_metric_anchors_unchanged=True,metric_geometry_sha256=sha(out/'metric_geometry.json'))
 (out/'input_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 report=dict(episode=manifest['episode'],input_dir=str(out),physical_scene=str(scene),physical_scene_sha256=sha(scene),records=records,immutable_original_adapter_files={n:sha(source/n) for n in ['tracking_reference.npz','reference_trajectory.npz','metric_geometry.json']},copied_files_equal=all(sha(source/n)==sha(out/n) for n in ['tracking_reference.npz','reference_trajectory.npz','metric_geometry.json']),not_an_official_ground_truth_reconstruction=True)
 (out/'repaired_mesh_export_validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--episode',type=int,required=True);p.add_argument('--run',required=True);p.add_argument('--output',required=True);prepare(**vars(p.parse_args()))
