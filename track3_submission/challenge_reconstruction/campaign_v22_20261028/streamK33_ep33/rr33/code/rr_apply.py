"""apply a solution to camera_from_object/world_from_object; shift px -> camera x,y translation at the spoon COM depth. helper for overlay and target build."""
import numpy as np
from scipy.spatial.transform import Rotation as Rot
def apply(z, scan, sol):
    names=[str(n) for n in z['names']]; k=names.index('wooden_spoon'); C=z['camera_from_object'][:,k].copy(); Wr=z['world_from_rgb']; fx,fy=574.9024658203125,574.8653564453125
    L,M,S,com=scan['long'],scan['mid'],scan['short'],scan['com']; N=len(C); W=z['world_from_object'][:,k].copy()
    for t in range(N):
        r,(m,s),(du,dv)=sol['roll'][t],sol['tilt'][t],sol['shift'][t]
        if r==0 and m==0 and s==0 and du==0 and dv==0: continue
        E=Rot.from_rotvec(L*np.radians(r)).as_matrix()@Rot.from_rotvec(M*np.radians(m)).as_matrix()@Rot.from_rotvec(S*np.radians(s)).as_matrix(); B=np.eye(4); B[:3,:3]=E; B[:3,3]=com-E@com
        T=C[t]@B; zz=(C[t][:3,:3]@com+C[t][:3,3])[2]; T[0,3]+=du*zz/fx; T[1,3]+=dv*zz/fy; C[t]=T; W[t]=Wr[t]@T
    return C,W,k
