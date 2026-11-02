"""Does the frozen first-object tape survive randomized physics, with no second hand changes at all?"""
import sys,json,numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
sys.path.insert(0,str(Path(__file__).parent))
from frontier_loop import Loop,spent_over
for spec in sys.argv[1:]:
    ep,plan,k1,side,k0=spec.split(',');e=Path(ep);t=Loop(e/'native_model.mjb',e/'baseline/physics.npz',Path(plan),side,int(k1));t.stage='real';t.set_rung('real');rng=np.random.default_rng(0);k0=int(k0);qi=t.s.object_qpos[k0];P=t.s.poses;i=P[0,k0];al=R.from_quat(i[3:7])*R.from_quat(i[10:14]).inv();gp=al.apply(P[:,k0,7:10])+i[:3]-al.apply(i[7:10]);gr=al*R.from_quat(P[:,k0,10:14]);n=len(P);print(e.name,'frozen object',k0,'frames',n)
    for label,rand in (('nominal',False),('rand0',True),('rand1',True),('rand2',True)):
        if rand:t.randomize(rng)
        else:
            t.randomize(rng);m=t.m;f,ma,ine,dt=t.nominal;m.geom_friction[:]=f;m.body_mass[:]=ma;m.body_inertia[:]=ine;m.opt.timestep=dt;t.s.substeps=int(round(1/(t.fps*dt)))
        t.s.reset(t.Q[0]);ep_=[];er=[];brk=None
        for f in range(n):
            t.s.step(t.T[f]);d=t.s.d;ep_.append(float(np.linalg.norm(d.qpos[qi:qi+3]-gp[f])));er.append(float((R.from_quat(d.qpos[qi+3:qi+7][[1,2,3,0]]).inv()*gr[f]).magnitude()))
            if brk is None and spent_over(ep_,er,t.frames):brk=f
        print(' ',label,'timestep',round(float(t.m.opt.timestep),5),'frozen object mean',round(100*np.mean(ep_),2),'cm',round(float(np.degrees(np.mean(er))),1),'deg | budget break at',brk,'| second-object onset',t.onset)
