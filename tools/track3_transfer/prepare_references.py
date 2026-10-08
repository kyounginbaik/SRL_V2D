from pathlib import Path
import json,shutil
import numpy as np,trimesh,pyarrow.parquet as pq
BASE=Path(__file__).resolve().parent.parent
report=[]
for ep in sorted(BASE.glob('episode_*')):
 files=list((ep/'retargeted').rglob('*.parquet'))
 if not files:continue
 if len(files)!=1:raise ValueError(f'Ambiguous retargeting output: {ep}')
 d=pq.read_table(files[0]).to_pylist()[0]
 assert d['fps']==20 and len(d['object_body_names'])==len(d['object_mesh_paths'])
 for name in d['object_body_names']:
  src=ep/'assets'/(name+'.obj');col=ep/'assets'/(name+'_collision.obj');urdf=ep/'assets'/(name+'.urdf')
  if not col.exists():
   mesh=trimesh.load(src,force='mesh');simple=mesh.simplify_quadric_decimation(face_count=4096) if len(mesh.faces)>4096 else mesh
   if not np.isfinite(simple.vertices).all():raise ValueError('Invalid collider')
   simple.export(col)
  text=urdf.read_text();old=f'<collision><geometry><mesh filename="{name}.obj"/>'
  if old in text:urdf.write_text(text.replace(old,f'<collision><geometry><mesh filename="{name}_collision.obj"/>'))
 shutil.copy2(files[0],ep/'reference.parquet')
 report.append({'episode':ep.name,'objects':d['object_body_names'],'frames':len(d['object_body_position']),'fps':d['fps'],'collision_faces_max':4096,'visual_meshes_unchanged':True})
(BASE/'reference_manifest.json').write_text(json.dumps(report,indent=2));print('Prepared',len(report),'references')
