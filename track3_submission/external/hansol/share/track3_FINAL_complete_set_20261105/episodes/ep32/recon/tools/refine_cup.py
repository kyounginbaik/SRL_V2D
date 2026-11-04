"""GT-free per-frame spoon pose refit for ep32 against video masks, stereo depth and optionally ball detections.
Pose p is compared with image p+1 (known 1-frame lag). usage: refine.py OUT.npz balls=0|1 [workers]"""
import sys,json,numpy as np,cv2,trimesh
from multiprocessing import Pool
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
C='/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/canonical/evaluation/episode_000032'
KD='/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK32_ep32/bundles/K32_R10/episode_000032'
K=json.load(open(f'{C}/intrinsics.json'));fx,fy,cx,cy,W,H=K['fx'],K['fy'],K['cx'],K['cy'],K['width'],K['height']
Z=np.load(f'{KD}/object_poses.npz');ks=list(Z['names']).index('blue_cup');T0=Z['camera_from_object'][:,ks]
J=np.load('/data1/SRL_V2D/challenge_submissions_SRL/cowork_ep32_kinematic_fixed_20261102/ep32_kinematic_fixed/ep32_fixed_object_poses.npz');TJ=J['camera_from_object'][:,list(J['names']).index('blue_cup')]
m=trimesh.load(f'{KD}/meshes/blue_cup.obj',force='mesh');v=np.asarray(m.vertices);f=np.asarray(m.faces);com=v.mean(0)
fc=v[f].mean(1);fn=np.asarray(m.face_normals);rs=np.random.default_rng(0).choice(len(f),min(800,len(f)),replace=False);fc=fc[rs];fn=fn[rs]
B=np.array([[-0.042,0.018,-0.048],[0.02,0.039,0.065]]);DET=json.load(open(f'{C}/ball_detections_blue_cup.json'))['frames']
USEB=sys.argv[2]=='balls=1';SC=0.5
def rd(p):return cv2.imread(p,-1)
def proj(p):return np.c_[fx*p[:,0]/p[:,2]+cx,fy*p[:,1]/p[:,2]+cy]
def sil(T):
    p=v@T[:3,:3].T+T[:3,3];u=(proj(p)*SC).round().astype(np.int32);I=np.zeros((int(H*SC),int(W*SC)),np.uint8)
    for t in f:cv2.fillConvexPoly(I,u[t],1)
    return I>0
def app(T,x):
    O=T.copy();E=R.from_rotvec(x[:3]*0.1).as_matrix();c=T[:3,:3]@com+T[:3,3];O[:3,:3]=E@T[:3,:3];O[:3,3]=E@(T[:3,3]-c)+c+x[3:]*0.01;return O
def load(i):
    o=f'{C}/objects/blue_cup/masks';Mk=rd(f'{o}/1/{i:06d}.png')>0;h=(rd(f'{o}/2/{i:06d}.png')>0)|(rd(f'{o}/3/{i:06d}.png')>0)|(rd(f'{C}/objects/wooden_spoon/masks/1/{i:06d}.png')>0)
    dv=rd(f'{C}/depth/{i:06d}.png').astype(np.float64);D=np.where(dv<65535,65535/np.maximum(dv,1)-1,np.nan)
    sz=(int(W*SC),int(H*SC));return cv2.resize(Mk.astype(np.uint8),sz,interpolation=cv2.INTER_NEAREST)>0,cv2.resize(h.astype(np.uint8),sz,interpolation=cv2.INTER_NEAREST)>0,D,DET[i] if i<len(DET) else []
def parts(T,Mk,h,D,det):
    S=sil(T);a=S&~h;b=Mk&~h;iou=(a&b).sum()/max((a|b).sum(),1);sc=((b&~a).sum()+(a&~b).sum())/max(b.sum(),1)
    pc=fc@T[:3,:3].T+T[:3,3];nc=fn@T[:3,:3].T;vis=(nc*pc).sum(1)<0;u=proj(pc[vis]).astype(int);ok=(u[:,0]>=0)&(u[:,0]<W)&(u[:,1]>=0)&(u[:,1]<H);u=u[ok];z=pc[vis][ok,2]
    dz=0.
    if len(u)>20:
        Dv=D[u[:,1],u[:,0]];keep=np.isfinite(Dv)&Mk[(u[:,1]*SC).astype(int),(u[:,0]*SC).astype(int)]&~h[(u[:,1]*SC).astype(int),(u[:,0]*SC).astype(int)]
        if keep.sum()>10:dz=float(np.median(np.abs(z[keep]-Dv[keep])))
    bl=np.nan
    if det:
        ub=proj(B@T[:3,:3].T+T[:3,3]);bl=float(np.median([min(30,np.min(np.linalg.norm(ub-np.asarray(d['uv']),axis=1))) for d in det]))
    return sc,iou,dz,bl
def cost(x,T,dat):
    O=app(T,x);sc,iou,dz,bl=parts(O,*dat);c=sc+min(dz,.05)/.02*.15+.002*(x**2).sum()
    if USEB and np.isfinite(bl):c+=bl/30*.5
    return c
def one(p):
    i=p+1;Mk,h,D,det=load(i);c=T0[p][:3,:3]@com+T0[p][:3,3];det=[d for d in det if np.linalg.norm(np.asarray(d['camera_point'])-c)<.15];dat=(Mk,h,D,det)
    best=None
    for s,Ts in enumerate((T0[p],TJ[p])):
        r=minimize(cost,np.zeros(6),args=(Ts,dat),method='Powell',bounds=[(-8,8)]*3+[(-6,6)]*3,options=dict(maxfev=500,xtol=.02,ftol=1e-3))
        if best is None or r.fun<best[0]:best=(r.fun,app(Ts,r.x),s)
    return p,best[1],best[2],parts(T0[p],*dat),parts(best[1],*dat),parts(TJ[p],*dat),len(det),int(Mk.sum())
if __name__=='__main__':
    N=len(T0)-1
    with Pool(int(sys.argv[3]) if len(sys.argv)>3 else 16) as P:res=P.map(one,range(N),chunksize=4)
    T=T0.copy();src=np.zeros(N,int);A=np.zeros((N,3,4));nd=np.zeros(N,int);mp=np.zeros(N,int)
    for p,Tb,s,a,b,c,n,mk in res:T[p]=Tb;src[p]=s;A[p,0]=a;A[p,1]=b;A[p,2]=c;nd[p]=n;mp[p]=mk
    np.savez(sys.argv[1],T=T,T0=T0,src=src,parts=A,nd=nd,mask=mp);print('DONE')
    print('mean iou init %.3f refit %.3f jisu %.3f'%(A[:,0,1].mean(),A[:,1,1].mean(),A[:,2,1].mean()),'ball px init %.1f refit %.1f jisu %.1f'%tuple(np.nanmean(A[:,j,3]) for j in range(3)))
