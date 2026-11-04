import sys, numpy as np
from nestlib import *
R='/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/canonical/evaluation/episode_000010'
P=np.load(sys.argv[1] if len(sys.argv)>1 else R+'/exports/final_v5/object_poses.npz')
names=[str(n) for n in P['names']]; W=P['world_from_object']
mb=load_mesh(R+'/objects/blue_cup/mesh.obj'); me=load_mesh(R+'/objects/beige_cup/mesh.obj')
ab,cb,_=container_axis(mb); ae,ce,_=container_axis(me)
pair=Pair(mb,me); ib,ie=names.index('blue_cup'),names.index('beige_cup')
vb=np.asarray(mb.vertices); hb=(vb-cb)@ab; ve=np.asarray(me.vertices); he=(ve-ce)@ae
rim_b=cb+ab*hb.max(); rim_e=ce+ae*he.max()
print('blue H',np.ptp(hb),'beige H',np.ptp(he))
for f in range(0,len(W),5):
    Tb,Te=W[f,ib],W[f,ie]
    axw=Tb[:3,:3]@ab
    rel=np.linalg.inv(Tb)@Te
    rimE=rel[:3,:3]@rim_e+rel[:3,3]   # beige rim centre in blue frame
    h=(rimE-cb)@ab - hb.max()          # beige rim height above blue rim along blue axis
    radial=np.linalg.norm((rimE-cb)-((rimE-cb)@ab)*ab)
    tilt=np.degrees(np.arccos(np.clip((rel[:3,:3]@ae)@ab,-1,1)))
    col=pair.collide(Tb,Te); d=pair.min_offset(Tb,Te,axw) if col else 0.0
    print(f'f{f:3d} rim-to-rim {h*100:6.2f} cm radial {radial*100:5.2f} tilt {tilt:5.1f} collide {int(col)} min_axial_offset {None if d is None else round(d*100,2)} cm')
