import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter
Z=np.load('../bundles/K32_R10/episode_000032/object_poses.npz');k=list(Z['names']).index('blue_cup');WR=Z['world_from_rgb']
def jit(T):
    r=R.from_matrix(T[:,:3,:3]);rate=np.degrees((r[:-1].inv()*r[1:]).magnitude());c=T[:,:3,3];sp=100*np.linalg.norm(np.diff(c,axis=0),axis=1);return rate,sp
for L in ('0.03','0.1','0.3'):
    h=np.load(f'cuphold_{L}.npz');Q=h['parts'];N=len(Q);ho=(np.arange(N)%4==0)&(np.arange(N)>=6);rest=(np.arange(N)%4!=0)&(np.arange(N)>=6)
    f=np.load(f'cupfull_{L}.npz');W=np.stack([WR[t]@f['T'][t] for t in range(len(f['T']))]);W0=np.stack([WR[t]@f['T0'][t] for t in range(len(f['T']))]);r,s=jit(W);r0,s0=jit(W0);Fq=f['parts']
    print(L,'heldout iou %.3f>%.3f min %.2f>%.2f | ball %.1f>%.1f | fit iou %.3f>%.3f min %.2f>%.2f'%(Q[ho,0,1].mean(),Q[ho,1,1].mean(),Q[ho,0,1].min(),Q[ho,1,1].min(),np.nanmean(Q[ho,0,3]),np.nanmean(Q[ho,1,3]),Q[rest,0,1].mean(),Q[rest,1,1].mean(),Q[rest,0,1].min(),Q[rest,1,1].min()))
    ch=np.degrees((R.from_matrix(W0[:,:3,:3]).inv()*R.from_matrix(W[:,:3,:3])).magnitude());cm=100*np.linalg.norm(W[:,:3,3]-W0[:,:3,3],axis=1)
    print('   full: iou %.3f>%.3f min %.2f>%.2f; step deg mean %.1f>%.1f max %.0f>%.0f; step cm mean %.2f>%.2f max %.1f>%.1f; change deg mean %.1f max %.0f, cm mean %.2f max %.1f'%(Fq[:,0,1].mean(),Fq[:,1,1].mean(),Fq[:,0,1].min(),Fq[:,1,1].min(),r0.mean(),r.mean(),r0.max(),r.max(),s0.mean(),s.mean(),s0.max(),s.max(),ch.mean(),ch.max(),cm.mean(),cm.max()))
