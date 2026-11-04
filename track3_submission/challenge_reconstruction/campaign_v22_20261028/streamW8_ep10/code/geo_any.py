import sys, json, numpy as np
from nestlib import *
root,pd=sys.argv[1],sys.argv[2]
P=np.load(pd+'/object_poses.npz'); names=[str(n) for n in P['names']]; W=P['world_from_object']
mj=json.loads(open(pd+'/meshes.json').read()); path=lambda n: mj[n] if isinstance(mj[n],str) else pd+'/'+mj[n]['file']
mb=load_mesh(path('blue_cup')); me=load_mesh(path('beige_cup'))
ab,cb,_=container_axis(mb); ae,ce,_=container_axis(me); hb=(np.asarray(mb.vertices)-cb)@ab; he=(np.asarray(me.vertices)-ce)@ae
ib,ie=names.index('blue_cup'),names.index('beige_cup'); pair=Pair(mb,me)
print('H blue',np.ptp(hb),'beige',np.ptp(he))
for f in range(0,len(W),int(sys.argv[3]) if len(sys.argv)>3 else 5):
    rel=np.linalg.inv(W[f,ib])@W[f,ie]; rim=rel[:3,:3]@(ce+ae*he.max())+rel[:3,3]; h=(rim-cb)@ab-hb.max()
    rad=np.linalg.norm((rim-cb)-((rim-cb)@ab)*ab); tilt=np.degrees(np.arccos(np.clip((rel[:3,:3]@ae)@ab,-1,1)))
    print(f'f{f:3d} h {h*100:6.2f} radial {rad*100:5.2f} tilt {tilt:5.1f} collide {int(pair.collide(W[f,ib],W[f,ie]))}')
