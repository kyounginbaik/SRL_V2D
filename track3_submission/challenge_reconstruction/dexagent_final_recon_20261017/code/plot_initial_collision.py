"""Cross-sections comparing exact source visuals and collision parts, no edits."""
from pathlib import Path
import json
import numpy as np,mujoco,trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
B=Path(__file__).resolve().parents[1];src=B.parent/'final_recon/episode_000001/assets'
m=mujoco.MjModel.from_xml_path(str(B/'runs/pilot_v2/episode_000001/seed_44/baseline/scene_physics.xml'))
d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
meshes={}
for o in ('white_pot','white_pot_lid'):
 b=m.body(o).id;tf=np.eye(4);tf[:3,:3]=d.xmat[b].reshape(3,3);tf[:3,3]=d.xpos[b]
 paths=[src/f'{o}_visual.obj',*sorted((src/f'{o}_convex').glob('*.obj'))]
 meshes[o]=[trimesh.load(p,force='mesh',process=False).apply_transform(tf) for p in paths]
fig,axs=plt.subplots(1,3,figsize=(15,5))
for ax,z in zip(axs,[.10,.11,.12]):
 for o,c in [('white_pot','#0059b2'),('white_pot_lid','#d33d15')]:
  for k,mesh in enumerate(meshes[o]):
   segments=trimesh.intersections.mesh_plane(mesh,[0,0,1],[0,0,z])
   if len(segments):ax.add_collection(LineCollection(segments[:,:,:2]*100,colors=c,linewidths=1.8 if k==0 else .7,alpha=1 if k==0 else .3))
 ax.set_aspect('equal');ax.set_xlim(-12,12);ax.set_ylim(-13,11);ax.grid(alpha=.2)
 ax.set_title(f'Initial world z={z:.2f}m');ax.set_xlabel('world x (cm)');ax.set_ylabel('world y (cm)')
fig.suptitle('Initial pot/lid intersections: blue=pot, red=lid; bold=source visual, faint=32 collision hulls each')
fig.tight_layout();fig.savefig(B/'refinement_20261017/visual_audit/episode_000001_initial_sections.png',dpi=160)

def cross2(a,b):return a[...,0]*b[...,1]-a[...,1]*b[...,0]
intersections=[]
for z in np.arange(.07,.14,.001):
 a=trimesh.intersections.mesh_plane(meshes['white_pot'][0],[0,0,1],[0,0,z])[:,:,:2]
 b=trimesh.intersections.mesh_plane(meshes['white_pot_lid'][0],[0,0,1],[0,0,z])[:,:,:2]
 if not len(a) or not len(b):continue
 p=a[:,None,0];q=b[None,:,0];r=a[:,None,1]-p;s=b[None,:,1]-q
 denom=cross2(r,s);safe=np.where(np.abs(denom)>1e-12,denom,np.nan)
 t=cross2(q-p,s)/safe;u=cross2(q-p,r)/safe
 valid=(t>1e-6)&(t<1-1e-6)&(u>1e-6)&(u<1-1e-6)
 points=(p+t[...,None]*r)[valid]
 if len(points):intersections.append(dict(world_z_m=float(z),count=len(points),points_xy_m=points.tolist()))
(B/'refinement_20261017/visual_audit/episode_000001_visual_intersections.json').write_text(json.dumps(dict(method='strict intersection of source visual triangle-mesh horizontal cross-section line segments',initial_pose_unchanged=True,planes_with_intersection=len(intersections),intersections=intersections),indent=2))
print('visual-mesh intersection planes:',len(intersections))
