import sys,json,time,numpy as np
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import yam_loop as L
for ep in sys.argv[1:]:
    T=L.YamTask(ep,'right',0);rng=np.random.default_rng(0);keep=rej=fail=0;scores=[];t0=time.monotonic();saved=L.NATURAL_PALM_MIN;L.NATURAL_PALM_MIN=-2.;left_n=[]
    for _ in range(60):
        g=T.propose(rng)
        if g is None:continue
        s=T.solve_hand(g,rng)
        if s is None:fail+=1;continue
        from scipy.spatial.transform import Rotation as R
        v=float(R.from_rotvec(s['r']).apply(T.local_normal)@T.natural_palm);scores.append(v);ok=s['residual_m']<.006 and s['penetration_m']<.006 and v>=saved;keep+=ok;rej+=(s['residual_m']<.006 and s['penetration_m']<.006 and not ok)
    print(json.dumps(dict(ep=Path(ep).name,local_normal=np.round(T.local_normal,2).tolist(),good_natural=int(keep),good_but_flipped=int(rej),no_solution=fail,palm_score_pct=np.percentile(scores,[10,50,90]).round(2).tolist() if scores else [],sec=round(time.monotonic()-t0))))
