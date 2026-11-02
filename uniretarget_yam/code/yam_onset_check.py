"""When each episode's goal really starts moving against the fixed earliest carry time, and how far it has gone by then."""
import sys
from pathlib import Path
import json,numpy as np
from scipy.spatial.transform import Rotation as R
FPS=20.
for e in sys.argv[1:]:
    z=np.load(Path(e)/'baseline/physics.npz');P=z['object_pose'][:,0]
    for k in range(P.shape[1]):
        o=P[:,k];i=o[0];al=R.from_quat(i[3:7])*R.from_quat(i[10:14]).inv();g=al.apply(o[:,7:10])+i[:3]-al.apply(i[7:10]);r=al*R.from_quat(o[:,10:14]);d=np.linalg.norm(g-g[0],axis=1);a=(r*r[0].inv()).magnitude();mv=(d>.015)|(a>.26);first=int(np.flatnonzero(mv)[0]) if mv.any() else -1;f=min(40,len(o)-1);v=np.linalg.norm(np.gradient(g,1/FPS,axis=0),axis=1)
        print(json.dumps(dict(episode=Path(e).name[-3:],object=k,frames=len(o),first_move_s=round(first/FPS,2),moved_by_2s_cm=round(100*float(d[f]),1),turned_by_2s_deg=round(float(np.degrees(a[f]))),peak_speed_ms=round(float(v.max()),2),peak_at_s=round(float(v.argmax()/FPS),2))))
