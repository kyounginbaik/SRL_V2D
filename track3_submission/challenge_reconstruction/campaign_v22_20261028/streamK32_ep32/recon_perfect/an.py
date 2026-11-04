import numpy as np
from scipy.spatial.transform import Rotation as R
for n in ('b0','b1'):
    d=np.load(f'refit_{n}.npz');A=d['parts'];T=d['T'];T0=d['T0'];N=len(A)
    rs=R.from_matrix(T[:,:3,:3]);rate=np.degrees((rs[:-1].inv()*rs[1:]).magnitude());r0=R.from_matrix(T0[:,:3,:3]);rate0=np.degrees((r0[:-1].inv()*r0[1:]).magnitude())
    ch=np.degrees((r0.inv()*rs).magnitude());dt=100*np.linalg.norm(T[:,:3,3]-T0[:,:3,3],axis=1)
    print(n,'iou init %.3f refit %.3f jisu %.3f | median %.3f %.3f %.3f | frames iou<.5: %d %d %d'%(A[:,0,1].mean(),A[:,1,1].mean(),A[:,2,1].mean(),np.median(A[:,0,1]),np.median(A[:,1,1]),np.median(A[:,2,1]),(A[:,0,1]<.5).sum(),(A[:,1,1]<.5).sum(),(A[:,2,1]<.5).sum()))
    has=np.isfinite(A[:,0,3]);print('  ball px on',has.sum(),'frames: init %.1f refit %.1f jisu %.1f; median %.1f %.1f %.1f'%(A[has,0,3].mean(),A[has,1,3].mean(),A[has,2,3].mean(),np.median(A[has,0,3]),np.median(A[has,1,3]),np.median(A[has,2,3])))
    print('  depth mm init %.1f refit %.1f'%(1e3*A[:,0,2].mean(),1e3*A[:,1,2].mean()))
    print('  change deg mean %.1f p95 %.1f max %.1f | cm mean %.2f p95 %.2f max %.2f'%(ch.mean(),np.percentile(ch,95),ch.max(),dt.mean(),np.percentile(dt,95),dt.max()))
    print('  rate deg/f mean %.1f>%.1f, frames>30: %d>%d, max %.0f>%.0f'%(rate0.mean(),rate.mean(),(rate0>30).sum(),(rate>30).sum(),rate0.max(),rate.max()))
    print('  src jisu share %.2f; worst refit iou frames'%d['src'].mean(),[(int(i),round(float(A[i,1,1]),2)) for i in np.argsort(A[:,1,1])[:14]])
    print('  still bad vs mask size',[int(d['mask'][i]) for i in np.argsort(A[:,1,1])[:14]])
