import sys,numpy as np
sys.argv=['r','x','balls=1'];exec(open('refine_cup.py').read().split("if __name__")[0]);SC=0.5
from scipy.signal import savgol_filter
ct=np.load(f'{C}/point_tracks_cotracker.npz',allow_pickle=True);k=np.flatnonzero(ct['object_ids']==0);TR=ct['tracks'][:,k];VIS=ct['visible'][:,k];d=np.load('mc2_0.npz');BP=d['BP'];ok=d['BPok']
def sg(T,wr,wt):
    O=T.copy();r=R.from_matrix(T[:,:3,:3]).as_rotvec();O[:,:3,:3]=R.from_rotvec(savgol_filter(r,wr,2,axis=0)).as_matrix();O[:,:3,3]=savgol_filter(T[:,:3,3],wt,2,axis=0);O[:6]=T[:6];return O
def ev(T):
    ious=[];sp=[];tr=[]
    for p in range(6,258):
        Mk,h,D,det=load(p+1);c=T[p][:3,:3]@com+T[p][:3,3];det=[q for q in det if np.linalg.norm(np.asarray(q['camera_point'])-c)<.25]
        sc,iou,dz,bl=parts(T[p],Mk,h,D,det);ious.append(iou);sp.append(bl)
        v=VIS[p+1]&ok;u=proj(BP[v]@T[p][:3,:3].T+T[p][:3,3]);tr.append(np.median(np.linalg.norm(u-TR[p+1][v],axis=1)) if v.sum()>3 else np.nan)
    r=R.from_matrix(T[:,:3,:3]);rv=r.as_rotvec();dj=np.degrees((R.from_rotvec(savgol_filter(rv,9,2,axis=0)).inv()*r).magnitude())[80:140]
    return np.mean(ious),np.nanmedian(sp),np.nanmedian(tr),dj.max(),dj.mean()
base=np.load('mc3_0.npz')['T']
for nm,T in (('lam0.4 as is',base),('lam0.4 sg rot9 pos7',sg(base,9,7))):
    i,s,t,jm,ja=ev(T);print(f'{nm:16s} iou {i:.3f} spike px {s:.1f} track px {t:.1f} | 4-7s jitter max {jm:.1f} mean {ja:.1f}',flush=True)
    np.save(f"cup_{nm.replace(' ','_').replace('+','')}.npy",T)
