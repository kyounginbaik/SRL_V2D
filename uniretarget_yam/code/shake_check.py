"""Is a shake survivor rigid? Report hand-relative slip and contacting digits through the rung."""
import sys,json,numpy as np,mujoco
from pathlib import Path
from scipy.spatial.transform import Rotation as R
sys.path.insert(0,str(Path(__file__).parent))
from frontier_loop import Loop,SHAKE_FRAMES
model,physics,plan,elites=sys.argv[1:5];t=Loop(Path(model),Path(physics),Path(plan),'right');t.set_rung('shake');m=t.m;rows=json.loads(Path(elites).read_text())
out=[]
for k,row in enumerate(rows[:12]):
    s=row['sample'];H=t.onset+SHAKE_FRAMES;Q,T=t.compile_tape(s,H);t.s.reset(Q[0]);ref=None;slip=[];track=[];digits={};table=0;f6=np.zeros(6)
    for f in range(H):
        t.s.step(T[f]);d=t.s.d;po=d.qpos[t.qo:t.qo+3].copy();ro=R.from_quat(d.qpos[t.qo+3:t.qo+7][[1,2,3,0]]);wr=R.from_matrix(d.xmat[t.body].reshape(3,3).copy());wp=d.xpos[t.body].copy()
        if f==t.onset+10:ref=(ro,wr,wr.inv().apply(po-wp))
        if ref is not None:
            slip.append((float(np.degrees(((ref[1].inv()*wr)*(ref[0].inv()*ro).inv()).magnitude())),1e3*float(np.linalg.norm(wr.inv().apply(po-wp)-ref[2]))));track.append(float(np.degrees((ro.inv()*t.gr_raw[f]).magnitude())))
            for i in range(d.ncon):
                c=d.contact[i];a,b=int(m.geom_bodyid[c.geom[0]]),int(m.geom_bodyid[c.geom[1]])
                if t.obj in (a,b):
                    o=a if b==t.obj else b
                    if o in t.right:n=m.body(o).name.replace('right_sharpa_wave_right_','');digits[n]=digits.get(n,0)+1
                    else:table+=1
    slip=np.array(slip);track=np.array(track);n=len(slip);out.append(row)
    print(k,'frontier',row['frontier'],'mean_deg',round(row['mean_deg'],1),'| slip deg max',round(float(slip[:,0].max()),0),'end',round(float(slip[-1,0]),0),'slip mm max',round(float(slip[:,1].max()),0),'| track deg max',round(float(track.max()),0),'mean',round(float(track.mean()),0),'| tableFrames',table,'| digits(frac)',{d_:round(v/n,2) for d_,v in sorted(digits.items(),key=lambda kv:-kv[1])[:5]})
