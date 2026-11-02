"""Does the goal path itself drive the object into its support? Lowest collider vertex per frame."""
import sys,numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
sys.path.insert(0,str(Path(__file__).parent))
from frontier_loop import Loop
for spec in sys.argv[1:]:
    model,phys,plan,k=spec.split(',');t=Loop(Path(model),Path(phys),Path(plan),'right',int(k));t.set_rung('real');local=t.r0.inv().apply(t.verts-t.o0);rest=float((t.verts@t.up).min());n=len(t.gp_raw)
    h=np.array([float(((t.gr_raw[f].apply(local)+t.gp_raw[f])@t.up).min())-rest for f in range(n)])*1e3;mv=np.linalg.norm(t.gp_raw-t.gp_raw[0],axis=1)
    bad=np.flatnonzero(h<-2);print(Path(model).parent.name,'obj',k,'frames',n,'goal lowest-vertex height mm: min',round(float(h.min()),1),'at',int(h.argmin()),'| frames below rest>2mm',len(bad),'| first',bad[:1].tolist(),'| sample',[(f,round(float(h[f]))) for f in range(t.onset,min(n,t.onset+70),8)])
