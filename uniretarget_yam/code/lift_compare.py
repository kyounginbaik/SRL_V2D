"""Statistics of fresh proposals on the lift rung: where does acquisition fail, per episode?"""
import sys,numpy as np,mujoco
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from frontier_loop import Loop
T=None;LIMIT=float(__import__('os').environ.get('SWEEP','.006'))
def init(e,side,k):
    global T;e=Path(e);T=Loop(e/'native_model.mjb',e/'baseline/physics.npz',e/'plan.npz',side,int(k));T.stage='lift';T.set_rung('lift')
def job(seed):
    t=T;m=t.m;rng=np.random.default_rng(seed);f6=np.zeros(6)
    tries=0
    for _ in range(200):
        g=t.propose(rng)
        if g is None:continue
        s=t.solve_hand(g,rng);tries+=1
        if s['residual_m']<.006 and s['penetration_m']<.006 and t.sweep(s)<LIMIT:break
    else:return None
    H=t.onset+25;Q,Tn=t.compile_tape(s,H);t.s.reset(Q[0]);digits=[];peak=0.;press=0.;jam=0.;lift=0.;wrist=0.;other=0.
    for f in range(H):
        t.s.step(Tn[f]);d=t.s.d;lift=max(lift,float((d.qpos[t.qo:t.qo+3]-t.o0)@t.up))
        if f<t.onset-6:continue
        hit=set();up=0.;F=0.;J=0.;O=0.
        for i in range(d.ncon):
            c=d.contact[i];a,b=int(m.geom_bodyid[c.geom[0]]),int(m.geom_bodyid[c.geom[1]]);mujoco.mj_contactForce(m,d,i,f6);fw=c.frame.reshape(3,3).T@f6[:3]
            if t.obj in (a,b):
                o=a if b==t.obj else b
                if o in t.right:hit.add(m.body(o).name.split('_')[-2]);up+=float((fw if b==t.obj else -fw)@t.up);F+=float(f6[0])
            elif a in t.right and b in t.right:J+=float(f6[0])
            elif a in t.right or b in t.right:O+=float(f6[0])
        digits.append(len(hit));peak=max(peak,up/t.mg);press=max(press,F/t.mg);jam=max(jam,J);other=max(other,O)
    return dict(tries=tries,digits=float(np.max(digits)),peak=peak,press=press,jam=jam,other=other,lift=1e3*lift,cov=s['coverage'],res=1e3*s['residual_m'])
for spec in sys.argv[1:]:
    e,side,k=spec.split(',')
    with Pool(4,init,(e,side,k)) as p:r=[x for x in p.map(job,range(32)) if x]
    def q(n):return np.percentile([x[n] for x in r],[50,90]).round(1).tolist()
    print(Path(e).name[-2:],'obj',k,side,'n',len(r),'digits',q('digits'),'upF/mg',q('peak'),'pressF/mg',q('press'),'selfJamN',q('jam'),'otherContactN',q('other'),'lift mm',q('lift'),'lift>40:',sum(x['lift']>40 for x in r),'cov',q('cov'),'solves per kept',q('tries'))
