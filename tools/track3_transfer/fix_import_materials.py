"""Avoid Newton splitting vertex colors into thousands of independent shapes.

Preserve visual vertices/faces, use a uniform URDF display material, and strip
color from collision-only OBJ files. Submission meshes remain untouched.
"""
from pathlib import Path
import json,os,xml.etree.ElementTree as ET
import numpy as np,trimesh
BASE=Path(os.environ.get('V2D_TRANSFER_RUN',Path(__file__).resolve().parent.parent))
report=[]
for path in BASE.glob('episode_*/assets/*.urdf'):
 tree=ET.parse(path);visual=tree.find('link/visual')
 mesh=trimesh.load(path.parent/visual.find('geometry/mesh').get('filename'),force='mesh')
 color=np.median(mesh.visual.vertex_colors[:,:3],axis=0)/255
 material=visual.find('material')
 if material is None:material=ET.SubElement(visual,'material',{'name':'object_uniform'})
 element=material.find('color')
 if element is None:element=ET.SubElement(material,'color')
 element.set('rgba',' '.join(map(str,[*color,1.])))
 tree.write(path,encoding='unicode',xml_declaration=True)
 collider=path.parent/tree.find('link/collision/geometry/mesh').get('filename')
 geometry=trimesh.load(collider,force='mesh')
 collider.write_text(trimesh.exchange.obj.export_obj(geometry,include_color=False))
 report.append({'urdf':str(path),'uniform_visual_rgb':color.tolist(),'visual_geometry_preserved':True,'collider_faces':len(geometry.faces)})
(BASE/'import_representation.json').write_text(json.dumps(report,indent=2))
print('Prepared',len(report),'import representations')
