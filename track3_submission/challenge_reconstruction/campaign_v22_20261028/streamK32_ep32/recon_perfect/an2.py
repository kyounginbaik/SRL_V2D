import numpy as np,glob
from scipy.spatial.transform import Rotation as R
for fn in ['full_0.1.npz']:
    try:d=np.load(fn)
    except Exception as e:print(fn,e);continue
    Q=d['parts'];T=d['T'];T0=d['T0'];N=len(Q);f=np.arange(N)>=35
    a=Q[:,0];b=Q[:,1]
    h=f&(np.arange(N)%3==1)&np.isfinite(a[:,3]);nh=f&(np.arange(N)%3!=1)&np.isfinite(a[:,3])
    rs=R.from_matrix(T[:,:3,:3]);rate=np.degrees((rs[:-1].inv()*rs[1:]).magnitude());r0=R.from_matrix(T0[:,:3,:3]);rate0=np.degrees((r0[:-1].inv()*r0[1:]).magnitude());ch=np.degrees((r0.inv()*rs).magnitude());dt=100*np.linalg.norm(T[:,:3,3]-T0[:,:3,3],axis=1)
    print(fn,'iou %.3f>%.3f min %.2f>%.2f <.5: %d>%d'%(a[f,1].mean(),b[f,1].mean(),a[f,1].min(),b[f,1].min(),(a[f,1]<.5).sum(),(b[f,1]<.5).sum()))
    print('   ball px held-out(%d) mean %.1f>%.1f med %.1f>%.1f | fitted(%d) mean %.1f>%.1f med %.1f>%.1f'%(h.sum(),a[h,3].mean(),b[h,3].mean(),np.median(a[h,3]),np.median(b[h,3]),nh.sum(),a[nh,3].mean(),b[nh,3].mean(),np.median(a[nh,3]),np.median(b[nh,3])))
    print('   rate mean %.1f>%.1f max %.0f>%.0f >30: %d>%d | change deg mean %.1f p95 %.1f cm mean %.2f max %.2f | depth mm %.1f>%.1f'%(rate0.mean(),rate.mean(),rate0.max(),rate.max(),(rate0>30).sum(),(rate>30).sum(),ch.mean(),np.percentile(ch,95),dt.mean(),dt.max(),1e3*a[f,2].mean(),1e3*b[f,2].mean()))
    if fn.startswith('full'):
        print('   worst steps',[(int(i),round(float(rate[i]),1)) for i in np.argsort(-rate)[:10]])
        print('   f89-90 151-153 iou',[(p,round(float(a[p,1]),2),round(float(b[p,1]),2)) for p in (89,90,151,152,153,177,178)])
