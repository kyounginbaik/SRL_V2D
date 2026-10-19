"""Avoid Newton allocating a separate mesh for every reconstructed vertex color."""
from pathlib import Path
import xml.etree.ElementTree as ET,json,hashlib,shutil,datetime
import numpy as np,trimesh
B=Path(__file__).resolve().parents[1]
reports=[]
for folder in [B/'frozen/episode_000001',B.parent/'final_recon_candidates/episode_000001_contact_tuned']:
 for name in ['white_pot','white_pot_lid']:
  source=folder/'assets'/f'{name}.obj';visual=source.with_name(name+'_visual.obj');lines=[]
  for line in source.read_text().splitlines():
   fields=line.split()
   if fields and fields[0]=='v':line=' '.join(fields[:4])
   if fields and fields[0] in ['mtllib','usemtl']:continue
   lines.append(line)
  visual.write_text('\n'.join(lines)+'\n')
  a=trimesh.load(source,force='mesh',process=False);b=trimesh.load(visual,force='mesh',process=False)
  assert np.array_equal(a.vertices,b.vertices) and np.array_equal(a.faces,b.faces)
  urdf=source.with_suffix('.urdf');root=ET.parse(urdf)
  for e in root.findall('.//visual/geometry/mesh'):e.set('filename',visual.name)
  ET.indent(root);root.write(urdf,encoding='utf-8',xml_declaration=True)
  reports.append({'folder':str(folder),'source':source.name,'runtime_visual':visual.name,'vertices':len(a.vertices),'faces':len(a.faces),'geometry_identical':True,'collision_meshes_unchanged':True})
manifest=json.loads((B/'frozen_manifest.json').read_text());shutil.copy2(B/'frozen_manifest.json',B/'frozen_manifest_before_visual_fix.json')
f=B/'frozen/episode_000001';manifest[0]['asset_sha256']={str(p.relative_to(f)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((f/'assets').rglob('*')) if p.is_file()};(B/'frozen_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');(B/'visual_mesh_fix.json').write_text(json.dumps(reports,indent=2)+'\n');print(json.dumps(reports,indent=2))
