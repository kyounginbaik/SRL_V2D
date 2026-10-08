"""Manual image-derived rack geometry. This module never reads reference meshes."""
import json
from pathlib import Path
import numpy as np
import trimesh
ROOT=Path(__file__).resolve().parent

def outline(length,width,r,n=8):
 points=[]
 for x,y,start in [(length/2-r,width/2-r,0),(-length/2+r,width/2-r,90),(-length/2+r,-width/2+r,180),(length/2-r,-width/2+r,270)]:
  for theta in np.deg2rad(np.linspace(start,start+90,n,endpoint=False)):
   points.append([x+r*np.cos(theta),y+r*np.sin(theta)])
 return np.array(points)

def vessel(length,width,height,wall,bottom,r):
 outer=outline(length,width,r)
 inner=outline(length-2*wall,width-2*wall,max(r-wall,.005))
 rings=[np.column_stack([outer,np.zeros(len(outer))]),np.column_stack([outer,np.full(len(outer),height)]),np.column_stack([inner,np.full(len(inner),height)]),np.column_stack([inner,np.full(len(inner),bottom)])]
 v=np.concatenate(rings); n=len(outer); faces=[]
 for a,b in [(0,1),(1,2),(2,3)]:
  for i in range(n):
   j=(i+1)%n
   faces.extend([[a*n+i,a*n+j,b*n+j],[a*n+i,b*n+j,b*n+i]])
 for ring,reverse in [(0,True),(3,False)]:
  center=len(v);v=np.vstack([v,rings[ring].mean(axis=0)])
  for i in range(n):
   tri=[center,ring*n+i,ring*n+(i+1)%n]
   faces.append(tri[::-1] if reverse else tri)
 mesh=trimesh.Trimesh(v,faces,process=True)
 mesh.fix_normals()
 return mesh

def build(width,peg_height):
 pieces=[vessel(1,width,.055,.012,.009,.055)]
 holder=vessel(.16,width-.025,.235,.015,.01,.06)
 holder.apply_translation([-.414,0,.009]);pieces.append(holder)
 for y in [-width*.24,width*.24]:
  for x in np.linspace(-.275,.245,6):
   peg=trimesh.creation.conical_frustum(radius_base=.016,radius_top=.010,height=peg_height,sections=16) if hasattr(trimesh.creation,'conical_frustum') else trimesh.creation.cylinder(radius=.013,height=peg_height,sections=16)
   # Cylinder primitive is centered at origin.
   peg.apply_translation([x,y,.009+peg_height/2]);pieces.append(peg)
 for x in np.linspace(-.26,.44,15):
  rib=trimesh.creation.box(extents=[.005,width-.045,.004]);rib.apply_translation([x,0,.011]);pieces.append(rib)
 mesh=trimesh.util.concatenate(pieces)
 mesh.visual.vertex_colors=[235,235,225,255]
 return mesh

if __name__=='__main__':
 variants=[]
 for width,height in [(.30,.115),(.34,.115),(.30,.14)]:
  out=ROOT/f'w{width:.2f}_h{height:.3f}';out.mkdir(exist_ok=True)
  build(width,height).export(out/'generated.glb')
  variants.append({'directory':str(out),'length':1,'width':width,'peg_height':height,'pegs_per_row':6})
 (ROOT/'provenance.json').write_text(json.dumps({'method':'Manual parametric reconstruction from public RGB image; no GT geometry used','image':'/data1/SRL_V2D/challenge_reconstruction/public_mesh_benchmark/mvsam3d/plastic_dish_rack_5v_tracked/input/images/000000.png','variants':variants},indent=2))
