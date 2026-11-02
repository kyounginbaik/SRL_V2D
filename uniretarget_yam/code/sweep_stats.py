import sys,time,numpy as np
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from frontier_loop import Loop
e=Path(sys.argv[1]);t=Loop(e/'native_model.mjb',e/'baseline/physics.npz',e/'plan.npz',sys.argv[2],int(sys.argv[3]));rng=np.random.default_rng(0);v=[];t0=time.monotonic();tp=ts=0.
while len(v)<60 and time.monotonic()-t0<70:
    a=time.monotonic();g=t.propose(rng)
    if g is None:continue
    s=t.solve_hand(g,rng);tp+=time.monotonic()-a
    if s['residual_m']>=.006 or s['penetration_m']>=.006:continue
    a=time.monotonic();v.append(t.sweep(s));ts+=time.monotonic()-a
v=np.array(v)*1e3;print(e.name[-2:],'kept before sweep',len(v),'sec/solve',round(tp/max(1,len(v)),2),'sec/sweep',round(ts/max(1,len(v)),4),'sweep mm pct10/50/90',np.percentile(v,[10,50,90]).round(1).tolist(),'<2mm',int((v<2).sum()),'<4mm',int((v<4).sum()),'<6mm',int((v<6).sum()),'<8',int((v<8).sum()))
