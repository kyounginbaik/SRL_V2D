"""Which rule ended the canonical frontier: object lost, angle spike, or budget?"""
import sys,numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
z=np.load(Path(sys.argv[1])/'physics.npz');T=int(sys.argv[2]);P=z['object_pose']
for w in range(P.shape[1]):
    o=P[:,w,0];i=o[0];al=R.from_quat(i[3:7])*R.from_quat(i[10:14]).inv();gp=al.apply(o[:,7:10])+i[:3]-al.apply(i[7:10]);gr=al*R.from_quat(o[:,10:14]);ep=np.linalg.norm(o[:,:3]-gp,axis=1);er=np.degrees((R.from_quat(o[:,3:7]).inv()*gr).magnitude());n=len(ep);f=np.arange(1,n+1);bp=.03*(.9*f+.1*T);br=30*(.9*f+.1*T)
    print('world',w,'n',n,'mean',round(100*ep.mean(),2),round(er.mean(),1),'| first pos>15cm',(np.flatnonzero(ep>.15)[:1]).tolist(),'first rot>150',(np.flatnonzero(er>150)[:1]).tolist(),'first posbudget',(np.flatnonzero(np.cumsum(ep)>bp)[:1]).tolist(),'first rotbudget',(np.flatnonzero(np.cumsum(er)>br)[:1]).tolist())
    print('  err by frame',[(int(k),round(100*float(ep[k]),1),int(er[k])) for k in range(100,n,12)])
