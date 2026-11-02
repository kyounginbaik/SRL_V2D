"""Is the fingertip error stopped-short or pushed-through? Signed along the surface normal, plus what touches."""
import sys,numpy as np,mujoco
from pathlib import Path
from scipy.spatial.transform import Rotation as R
sys.path.insert(0,str(Path(__file__).parent))
from frontier_loop import Loop
e=Path(sys.argv[1]);t=Loop(e/'native_model.mjb',e/'baseline/physics.npz',e/'plan.npz',sys.argv[2],int(sys.argv[3]));t.stage='lift';t.set_rung('lift');m=t.m;rng=np.random.default_rng(4);n=0;f6=np.zeros(6);rows=[]
while n<8:
    g=t.propose(rng)
    if g is None:continue
    s=t.solve_hand(g,rng)
    if s['residual_m']>=.006 or s['penetration_m']>=.006 or t.sweep(s)>=.006:continue
    n+=1;H=t.onset+1;Q,T=t.compile_tape(s,H);t.s.reset(Q[0])
    for f in range(H):t.s.step(T[f])
    d=t.s.d;pos=d.qpos[t.qo:t.qo+3];mv=R.from_quat(d.qpos[t.qo+3:t.qo+7][[1,2,3,0]])*t.r0.inv();out={}
    for x in ('thumb','index','middle'):
        nrm=mv.apply(np.asarray(g['n_'+x]));goal=pos+mv.apply(np.asarray(g[x])+.007*np.asarray(g['n_'+x])-t.o0);err=t.tip_world(d,x)-goal;out[x]=(round(1e3*float(err@nrm),1),round(1e3*float(np.linalg.norm(err-(err@nrm)*nrm)),1))
    touch={};jam={}
    for i in range(d.ncon):
        c=d.contact[i];a,b=int(m.geom_bodyid[c.geom[0]]),int(m.geom_bodyid[c.geom[1]]);mujoco.mj_contactForce(m,d,i,f6);nm=lambda q:m.body(q).name.split('_')[-2]+m.body(q).name.split('_')[-1] if q in t.right else m.body(q).name[:7]
        if t.obj in (a,b) and (a in t.right or b in t.right):o=a if b==t.obj else b;touch[nm(o)]=round(touch.get(nm(o),0)+float(f6[0]),1)
        elif a in t.right and b in t.right:k=nm(a)+'|'+nm(b);jam[k]=round(jam.get(k,0)+float(f6[0]),1)
    print(n,'press',round(1e3*s.get('press',0),1),'(normal+out,tangent) mm',out,'objTouch',touch,'selfTop',sorted(jam.items(),key=lambda kv:-kv[1])[:2])
