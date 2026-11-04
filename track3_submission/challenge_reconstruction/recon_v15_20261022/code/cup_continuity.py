"""Opening-axis continuity for axially symmetric cups (spin is unscored, so the only discrete ambiguity left is upside-down).
Per frame two hypotheses: the tracked pose and its upside-down twin (180 deg about a body axis perpendicular to the revolution
axis, through the revolution centre, so the cup occupies the same place). Viterbi picks the sequence minimising
sum_k (angle between consecutive world opening axes / theta0)^2 + lam * [twin used]. Frame 0 is the tracked pose."""
import numpy as np, trimesh
from scipy.spatial.transform import Rotation as R
def body_axis(v):
    c=v-v.mean(0); w,U=np.linalg.eigh(c.T@c); best=None
    for i in range(3):
        a=U[:,i]; h=c@a; r=np.linalg.norm(c-np.outer(h,a),axis=1); b=np.digitize(h,np.linspace(h.min(),h.max(),20))
        s=np.mean([np.std(r[b==k]) for k in np.unique(b) if (b==k).sum()>20]); best=min(best,(s,a),key=lambda t:t[0]) if best else (s,a)
    a=best[1]; h=c@a; r=np.linalg.norm(c-np.outer(h,a),axis=1)
    lo=h<np.percentile(h,8); hi=h>np.percentile(h,92)
    if np.median(r[hi])<np.median(r[lo]): a=-a          # opening (wider rim) along +a
    # pivot: point on the revolution axis at mid-height (mid of the extent along the axis), so the twin occupies the same volume
    m=v.mean(0); h=(v-m)@a; return a, m+a*0.5*(h.max()+h.min())
def twin_transform(a, c):
    p=np.cross(a,[1,0,0]); p=p if np.linalg.norm(p)>1e-3 else np.cross(a,[0,1,0]); p/=np.linalg.norm(p)
    F=np.eye(4); F[:3,:3]=R.from_rotvec(np.pi*p).as_matrix(); F[:3,3]=c-F[:3,:3]@c; return F
def fix(T, mesh_vertices, theta0=15.0, lam=0.05):
    a,c=body_axis(np.asarray(mesh_vertices,float)); F=twin_transform(a,c); n=len(T)
    H=np.stack([T, T@F])                                    # (2,n,4,4)
    ax=np.einsum('hnij,j->hni',H[:,:,:3,:3],a)             # world opening axes
    cost=np.full((n,2),np.inf); back=np.zeros((n,2),int); cost[0]=[0,np.inf]
    for k in range(1,n):
        for s in range(2):
            ang=np.degrees(np.arccos(np.clip(ax[:,k-1]@ax[s,k],-1,1)))
            tot=cost[k-1]+(ang/theta0)**2+lam*s; back[k,s]=int(np.argmin(tot)); cost[k,s]=tot[back[k,s]]
    s=int(np.argmin(cost[-1])); path=np.zeros(n,int)
    for k in range(n-1,-1,-1): path[k]=s; s=back[k,s]
    return H[path,np.arange(n)], path
