"""Mesh-mesh interpenetration census over all clips (no GT). Per pair: colliding frames, and frames where the smaller
object's centroid lies inside the larger object's bounding cylinder (nested candidates)."""
import json, sys, numpy as np
from pathlib import Path
from nestlib import *
W=Path('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026')
clips=[(r,r/'exports/r8_base') for r in sorted((W/'canonical/public').glob('episode_*'))+sorted((W/'canonical/public_heldout').glob('episode_*'))]
clips+=[(W/'canonical/evaluation'/e.name, e) for e in sorted((W/'selected20_final_v5').glob('episode_*'))]
out={}
for root,pd in clips:
    P=np.load(pd/'object_poses.npz'); names=[str(n) for n in P['names']]; Wf=P['world_from_object']
    if len(names)<2: continue
    mj=json.loads((pd/'meshes.json').read_text()); mj={k:(v if v.startswith('/') else str(pd/v)) if isinstance(v,str) else str(pd/v['file']) for k,v in mj.items()}
    ms=[load_mesh(mj[n]) for n in names]
    vol=[np.prod(m.extents) for m in ms]; o,i=(0,1) if vol[0]>=vol[1] else (1,0)
    pair=Pair(ms[o],ms[i]); ax,c,_=container_axis(ms[o]); v=np.asarray(ms[o].vertices); h=(v-c)@ax; r=np.linalg.norm((v-c)-np.outer(h,ax),axis=1)
    ci=ms[i].centroid
    col=[];nest=[]
    for f in range(len(Wf)):
        col.append(pair.collide(Wf[f,o],Wf[f,i]))
        rel=np.linalg.inv(Wf[f,o])@Wf[f,i]; p=rel[:3,:3]@ci+rel[:3,3]-c; hh=p@ax; rr=np.linalg.norm(p-hh*ax)
        nest.append(bool(h.min()<hh<h.max() and rr<np.percentile(r,90)))
    col=np.array(col); nest=np.array(nest)
    key=f'{root.parent.name}/{root.name}'; out[key]=dict(outer=names[o],inner=names[i],frames=len(Wf),collide=int(col.sum()),nested=int(nest.sum()),collide_nested=int((col&nest).sum()),
        collide_nested_frames=[int(x) for x in np.nonzero(col&nest)[0]])
    print(key,names[o],'>',names[i],'frames',len(Wf),'collide',col.sum(),'nested',nest.sum(),'both',(col&nest).sum(),flush=True)
json.dump(out,open('../out/census_v5.json','w'),indent=1)
