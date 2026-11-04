import numpy as np,json,re
from scipy.spatial.transform import Rotation as R
names=['old','hansol','ours1','merged']
Z=np.load('../bundles/K32_R10/episode_000032/object_poses.npz');WR=Z['world_from_rgb']
exec(open('marks.py').read().split("best=None")[0].split("RH=ref")[0]) if 0 else None
for o in ('spoon','cup'):
    for h in (1,0):
        d=np.load(f'mg_{o}_{h}.npz');Q=d['parts'];T=d['T'];N=len(Q);A=6 if o=='cup' else 35;f=np.arange(N)>=A
        print(o,'HOLD' if h else 'FULL','spikes',len(d['B']))
        print('   iou mean',np.round(Q[f,:,1].mean(0),3),'min',np.round(Q[f,:,1].min(0),2),'<.5',(Q[f,:,1]<.5).sum(0),'depth mm',np.round(1e3*Q[f,:,2].mean(0),1))
        ho=f&(np.arange(N)%3==1)&np.isfinite(Q[:,0,3]);print('   spike px (old 3D-model fit by each track run) heldout frames',ho.sum(),np.round(np.nanmean(Q[ho,:,3],axis=0),1),'median',np.round(np.nanmedian(Q[ho,:,3],axis=0),1))
        W=np.einsum('nij,njk->nik',WR,T);r=R.from_matrix(W[:,:3,:3]);q=np.degrees((r[:-1].inv()*r[1:]).magnitude());c=100*np.linalg.norm(np.diff(W[:,:3,3],axis=0),axis=1);print('   step deg mean %.1f max %.0f >15:%d | step cm mean %.2f max %.2f'%(q.mean(),q.max(),(q>15).sum(),c.mean(),c.max()))
