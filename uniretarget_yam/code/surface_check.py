"""Are sampled surface points actually on the collider? Independent check with point-to-hull distance."""
import sys,numpy as np
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from frontier_loop import Loop
for spec in sys.argv[1:]:
    e,side,k=spec.split(',');e=Path(e);t=Loop(e/'native_model.mjb',e/'baseline/physics.npz',e/'plan.npz',side,int(k));P,N=t.surface(np.random.default_rng(0),1500);best=[]
    for p in P[:300]:
        d=np.inf
        for g in t.geoms:
            pg,Rg=t.rest_geoms[int(g)];eq=t.hulls[int(g)];d=min(d,float(np.max(((p-pg)@Rg)@eq[:,:3].T+eq[:,3])))
        best.append(d)
    b=np.abs(best)*1e3;print(e.name[-2:],'obj',k,'kept',len(P),'of 1500 | distance to collider surface mm pct50/90/max',np.percentile(b,[50,90,100]).round(2).tolist())
