"""Can each hand actually collide with each object in the exported model?"""
import sys,numpy as np,mujoco
from pathlib import Path
for e in map(Path,sys.argv[1:]):
    m=mujoco.MjModel.from_binary_path(str(e/'native_model.mjb'));d=mujoco.MjData(m);free=[j for j in range(m.njnt) if m.jnt_type[j]==0];ex={(int(b>>16),int(b&0xffff)) for b in m.exclude_signature}
    z=np.load(e/'baseline/physics.npz');q0=z['native_qpos'][0,0];P=z['object_pose'][0,0];print(e.name[-2:],'free joints',len(free),'task objects',P.shape[0],'nexclude',m.nexclude)
    for j in free:
        b=int(m.jnt_bodyid[j]);qi=int(m.jnt_qposadr[j]);g=np.flatnonzero(m.geom_bodyid==b);k=int(np.argmin(np.linalg.norm(P[:,:3]-q0[qi:qi+3],axis=1)));row={}
        for side in ('left','right'):
            hand=[x for x in range(m.ngeom) if m.body(int(m.geom_bodyid[x])).name.startswith(side+'_') and m.geom_contype[x]+m.geom_conaffinity[x]>0];ok=bl=ex_=0
            for h in hand:
                for o in g:
                    pair=(m.geom_contype[h]&m.geom_conaffinity[o])|(m.geom_contype[o]&m.geom_conaffinity[h])
                    x=tuple(sorted((int(m.geom_bodyid[h]),b)));y=x in ex or x[::-1] in ex
                    if not pair:bl+=1
                    elif y:ex_+=1
                    else:ok+=1
            row[side]=dict(enabled=ok,bitmask_off=bl,excluded=ex_)
        print('  body',b,m.body(b).name[:26],'task obj',k,'geoms',len(g),'types',sorted({int(m.geom_type[x]) for x in g}),'contype/aff',int(m.geom_contype[g[0]]),int(m.geom_conaffinity[g[0]]),'mass',round(float(m.body_mass[b]),3),'qpos0dist',round(float(np.linalg.norm(q0[qi:qi+3]-P[k,:3])),4),row)
