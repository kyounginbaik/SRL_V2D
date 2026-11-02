"""Does the fast sim's randomized pass rate predict official GPU MP-SR on tapes we already scored?"""
import sys,json,numpy as np
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from frontier_loop import Loop,LOST_P,LOST_R
T=None
def init(e,plan,side,k,pending):
    global T;e=Path(e);T=Loop(e/'native_model.mjb',e/'baseline/physics.npz',Path(plan),side,int(k));T.stage='real';T.set_rung('real');T.pending=set(pending);T.allow_park=False
def job(a):
    s,seed,mode=a;rng=np.random.default_rng(seed)
    if mode=='rand':T.randomize(rng);j=rng.uniform(-.003,.003,3)*np.array([1,1,0])
    else:
        if hasattr(T,'nominal'):
            m=T.m;f,ma,ine,dt=T.nominal;m.geom_friction[:]=f;m.body_mass[:]=ma;m.body_inertia[:]=ine;m.opt.timestep=dt;T.s.substeps=int(round(1/(T.fps*dt)))
        j=rng.uniform(-.001,.001,3)*np.array([1,1,0])
    s2={k:v for k,v in s.items() if k!='cap'};r=T.trial(s2,T.frames-1,j);return r['frontier'],r['mean_cm'],r['mean_deg'],r['frames']
for spec in sys.argv[1:]:
    label,e,run,plan,side,k,official=spec.split(',');s=json.loads((Path(run)/'canonical_pass.json').read_text())['sample']
    for mode in ('rand','light'):
        with Pool(4,init,(e,plan,side,k,[])) as p:r=p.map(job,[(s,i,mode) for i in range(24)])
        full=sum(x[0]>=max(y[3] for y in r)-1 for x in r);c=np.array([x[1] for x in r]);d=np.array([x[2] for x in r]);ok=((c<3)&(d<30)&np.array([x[0]>=max(y[3] for y in r)-1 for x in r])).mean()
        print(label,'official',official,mode,'pass',round(float(ok),2),'full-length',full,'/24','cm med/p90',np.percentile(c,[50,90]).round(2).tolist(),'deg med/p90',np.percentile(d,[50,90]).round(1).tolist())
