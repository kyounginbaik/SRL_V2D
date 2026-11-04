import numpy as np
from scipy.spatial.transform import Rotation as Rot
def shift(T,d):
    F=len(T); t=np.clip(np.arange(F)+d,0,F-1); i0=np.floor(t).astype(int); i1=np.minimum(i0+1,F-1); a=t-i0
    out=T.copy(); out[:,:3,3]=(1-a)[:,None]*T[i0,:3,3]+a[:,None]*T[i1,:3,3]
    R0=Rot.from_matrix(T[i0,:3,:3]); R1=Rot.from_matrix(T[i1,:3,:3]); rv=(R0.inv()*R1).as_rotvec()
    out[:,:3,:3]=(R0*Rot.from_rotvec(rv*a[:,None])).as_matrix(); return out
