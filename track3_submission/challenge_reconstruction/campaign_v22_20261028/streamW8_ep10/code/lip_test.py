"""Is the ep10 v5 deep-nest interpenetration confined to the blue rim lip band? (no GT)"""
import numpy as np, trimesh
from nestlib import *
R='/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/canonical/evaluation/episode_000010'
P=np.load(R+'/exports/final_v5/object_poses.npz'); names=[str(n) for n in P['names']]; W=P['world_from_object']
mb=load_mesh(R+'/objects/blue_cup/mesh.obj'); me=load_mesh(R+'/objects/beige_cup/mesh.obj'); ab,cb,_=container_axis(mb)
h=(np.asarray(mb.vertices)-cb)@ab; top=h.max()
for cut in [0.0,0.01,0.015,0.02,0.025]:
    keep=np.all(h[mb.faces]<top-cut,axis=1) if cut>0 else np.ones(len(mb.faces),bool)
    sub=trimesh.Trimesh(mb.vertices,mb.faces[keep],process=False); pr=Pair(sub,me)
    fr=list(range(104,161,4)); c=[pr.collide(W[f,0],W[f,1]) for f in fr]
    print(f'blue faces within {cut*100:.1f} cm of rim removed: colliding nested frames {sum(c)}/{len(c)}')
from scipy.spatial.transform import Rotation as Rot
ae,ce,_=container_axis(me); he=(np.asarray(me.vertices)-ce)@ae
def coax(Tb,Te,hh):
    rel=np.linalg.inv(Tb)@Te; a=rel[:3,:3]@ae; Ra=Rot.align_vectors([ab],[a])[0].as_matrix(); rel=rel.copy(); rel[:3,:3]=Ra@rel[:3,:3]
    rim=rel[:3,:3]@(ce+ae*he.max()); rel[:3,3]=cb+ab*(top+hh)-rim; return Tb@rel
for hh in [0.025,0.03,0.04,0.05,0.06]:
  for cut in [0.0,0.015,0.025]:
    keep=np.all(h[mb.faces]<top-cut,axis=1) if cut>0 else np.ones(len(mb.faces),bool)
    sub=trimesh.Trimesh(mb.vertices,mb.faces[keep],process=False); pr=Pair(sub,me)
    c=[pr.collide(W[f,0],coax(W[f,0],W[f,1],hh)) for f in range(104,161,8)]
    print(f'coax h={hh*100:.1f} cut {cut*100:.1f}: colliding {sum(c)}/{len(c)}')
