import numpy as np,json
from scipy.spatial.transform import Rotation as R
Z=np.load('../bundles/K32_R10/episode_000032/object_poses.npz');WR=Z['world_from_rgb'];names=['old','hansol','ours1','merged']
for o,A in (('spoon',35),('cup',6)):
    d=np.load(f'mg_{o}_0.npz');Q=d['parts'];T=d['T'];N=len(Q);f=np.arange(N)>=A
    print('==',o,'spikes',len(d['B']))
    print('  iou mean',dict(zip(names,np.round(Q[f,:,1].mean(0),3))),'min',np.round(Q[f,:,1].min(0),2),'frames<.5',(Q[f,:,1]<.5).sum(0).tolist())
    print('  depth mm',np.round(1e3*Q[f,:,2].mean(0),1).tolist(),' spike px vs each run\'s model (median)',np.round(np.nanmedian(Q[f,:,3],axis=0),1).tolist())
    for w,(a,b) in (('4.25-4.75s',(85,96)),('6.25-8.25s',(125,165)),('8.5-9.5s',(170,190))):
        r=np.arange(a,b);print(f'  {w}: iou',np.round(Q[r,:,1].mean(0),2).tolist(),'spike px',np.round(np.nanmedian(Q[r,:,3],axis=0),1).tolist())
    W=np.einsum('nij,njk->nik',WR,T);r=R.from_matrix(W[:,:3,:3]);q=np.degrees((r[:-1].inv()*r[1:]).magnitude());c=100*np.linalg.norm(np.diff(W[:,:3,3],axis=0),axis=1)
    print('  merged step deg mean %.1f max %.0f >15:%d | step cm mean %.2f max %.2f'%(q.mean(),q.max(),(q>15).sum(),c.mean(),c.max()))
    np.save(f'merged_{o}.npy',T)
