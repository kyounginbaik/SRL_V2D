import sys,json,numpy as np,cv2,trimesh
from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter
C='/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/canonical/evaluation/episode_000032'
KD='/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK32_ep32/bundles/K32_R10/episode_000032'
K=json.load(open(f'{C}/intrinsics.json'));fx,fy,cx,cy,W,H=K['fx'],K['fy'],K['cx'],K['cy'],K['width'],K['height']
Z=np.load(f'{KD}/object_poses.npz');k=list(Z['names']).index('blue_cup');Tc=Z['camera_from_object'][:,k];Tw=Z['world_from_object'][:,k];N=len(Tc)
J=np.load('/data1/SRL_V2D/challenge_submissions_SRL/cowork_ep32_kinematic_fixed_20261102/ep32_kinematic_fixed/ep32_fixed_object_poses.npz');TJ=J['camera_from_object'][:,list(J['names']).index('blue_cup')]
m=trimesh.load(f'{KD}/meshes/blue_cup.obj',force='mesh');v=np.asarray(m.vertices);f=np.asarray(m.faces);com=v.mean(0)
DET=json.load(open(f'{C}/ball_detections_blue_cup.json'))['frames']
def fit_balls():
    P=[]
    for p in range(N-1):
        for d in (DET[p+1] if p+1<len(DET) else []):
            x=np.asarray(d['camera_point']);b=Tc[p][:3,:3].T@(x-Tc[p][:3,3])
            if np.linalg.norm(b-com)<.15:P.append(b)
    P=np.array(P);cl=[]
    left=np.ones(len(P),bool)
    while left.sum()>20 and len(cl)<8:
        q=P[left];cnt=[(np.linalg.norm(q-x,axis=1)<.012).sum() for x in q[::3]];c=q[::3][int(np.argmax(cnt))]
        for _ in range(5):c=np.median(q[np.linalg.norm(q-c,axis=1)<.012],axis=0)
        n=(np.linalg.norm(P-c,axis=1)<.015)
        if (n&left).sum()<25:break
        cl.append(c);left&=~n
    print('cup balls body',len(cl),np.round(cl,3).tolist(),'points',len(P));return np.array(cl)
def proj(p):return np.c_[fx*p[:,0]/p[:,2]+cx,fy*p[:,1]/p[:,2]+cy]
B=fit_balls()
def sil(T):
    u=proj(v@T[:3,:3].T+T[:3,3]).round().astype(np.int32);I=np.zeros((H,W),np.uint8)
    for t in f:cv2.fillConvexPoly(I,u[t],1)
    return I>0
def rd(p):return cv2.imread(p,-1)
def iou(T,t):
    M=rd(f'{C}/objects/blue_cup/masks/1/{t:06d}.png')>0;h=(rd(f'{C}/objects/blue_cup/masks/2/{t:06d}.png')>0)|(rd(f'{C}/objects/blue_cup/masks/3/{t:06d}.png')>0);S=sil(T)>0;a=S&~h;b=M&~h;return (a&b).sum()/max((a|b).sum(),1)
def ball(T,t):
    det=DET[t] if t<len(DET) else []
    c=T[:3,:3]@com+T[:3,3];det=[d for d in det if np.linalg.norm(np.asarray(d['camera_point'])-c)<.15]
    if not det:return np.nan
    ub=proj(B@T[:3,:3].T+T[:3,3]);return float(np.median([min(30,np.min(np.linalg.norm(ub-np.asarray(d['uv']),axis=1))) for d in det]))
def jit(T):
    r=R.from_matrix(T[:,:3,:3]);rv=np.unwrap(r.as_rotvec(),axis=0);ref=savgol_filter(rv,11,2,axis=0);d=np.degrees((R.from_rotvec(ref).inv()*r).magnitude());c=T[:,:3,:3]@com+T[:,:3,3];dc=100*np.linalg.norm(c-savgol_filter(c,11,2,axis=0),axis=1)
    rate=np.r_[0,np.degrees((r[:-1].inv()*r[1:]).magnitude())];sp=np.r_[0,100*np.linalg.norm(np.diff(c,axis=0),axis=1)];return d,dc,rate,sp
dw,cw,rw,sw=jit(Tw);dJ,cJ,rJ,sJ=jit(np.stack([Z['world_from_rgb'][t]@TJ[t] for t in range(N)]))
I=np.array([iou(Tc[p],p+1) for p in range(N-1)]);IJ=np.array([iou(TJ[p],p+1) for p in range(N-1)]);Bl=np.array([ball(Tc[p],p+1) for p in range(N-1)])
np.savez('cup_metrics.npz',dw=dw,cw=cw,rw=rw,sw=sw,I=I,IJ=IJ,Bl=Bl)
print('world cup: jitter deg mean %.2f p95 %.2f max %.1f | cm mean %.2f p95 %.2f max %.2f | step deg mean %.2f max %.1f, cm mean %.2f max %.2f'%(dw.mean(),np.percentile(dw,95),dw.max(),cw.mean(),np.percentile(cw,95),cw.max(),rw.mean(),rw.max(),sw.mean(),sw.max()))
print('jisu       jitter deg mean %.2f p95 %.2f max %.1f | cm mean %.2f max %.2f'%(dJ.mean(),np.percentile(dJ,95),dJ.max(),cJ.mean(),cJ.max()))
print('iou mean %.3f min %.2f jisu %.3f; ball px mean %.1f med %.1f on %d'%(I.mean(),I.min(),IJ.mean(),np.nanmean(Bl),np.nanmedian(Bl),np.isfinite(Bl).sum()))
print('sec | jitterdeg max, jittercm max, step deg max, step cm max, iou min, ball max')
for s in range(0,N,10):
    e=min(s+10,N-1);print(f'{s/20:5.1f}s {dw[s:e].max():5.1f} {cw[s:e].max():5.2f} {rw[s:e].max():5.1f} {sw[s:e].max():5.2f} {I[s:e].min():5.2f} {np.nanmax(Bl[s:e]) if np.isfinite(Bl[s:e]).any() else float("nan"):5.1f}')
