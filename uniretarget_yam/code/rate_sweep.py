"""How fast can verified rigid grasps turn before slipping, on the real goal? No edits, nominal physics."""
import sys,json,numpy as np
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from frontier_loop import Loop
model,phys,plan,el=sys.argv[1:5];rows=json.loads(Path(el).read_text())[:30];H=175;T=None
def job(a):
    global T
    if T is None:T=Loop(Path(model),Path(phys),Path(plan),'right',0);T.stage='real';T.set_rung('real');T.rigid=True
    s,v,w,grip=a;base=dict(s,vmax=v,wmax=w)
    if grip:base['press_by']={k:base.get('press_by',{}).get(k,base['press'])*grip for k in base['normals']}
    T.rigid=False;Q,Tt=Loop.compile_tape(T,base,H);T.rigid=True;T.compile_tape_saved=None
    orig=T.compile_tape;T.compile_tape=lambda s_,h:(Q,Tt);r=T.trial(base,H,np.zeros(3));T.compile_tape=orig;return dict(v=v,w=w,grip=grip,frontier=r['frontier'],hold=r['hold'],lift=r['lift_mm'],cm=r['mean_cm'],deg=r['mean_deg'])
jobs=[(r['sample'],v,w,g) for r in rows for v,w in ((.15,.5),(.2,1.),(.3,2.),(.4,3.),(.6,5.),(1.,8.)) for g in (1.,2.)]
with Pool(4) as p:out=p.map(job,jobs,chunksize=2)
print('v w grip | frontier median/p90/max | hold med | lift med | cm deg med')
for v,w in ((.15,.5),(.2,1.),(.3,2.),(.4,3.),(.6,5.),(1.,8.)):
    for g in (1.,2.):
        x=[o for o in out if o['v']==v and o['w']==w and o['grip']==g];f=np.array([o['frontier'] for o in x]);print(v,w,g,'|',int(np.median(f)),int(np.percentile(f,90)),int(f.max()),'|',int(np.median([o['hold'] for o in x])),'|',int(np.median([o['lift'] for o in x])),'|',round(float(np.median([o['cm'] for o in x])),1),round(float(np.median([o['deg'] for o in x])),0))
