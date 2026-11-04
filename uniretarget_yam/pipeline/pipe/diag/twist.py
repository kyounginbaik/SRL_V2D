import numpy as np,json
from scipy.spatial.transform import Rotation as R
root='/uniretarget_yam/episodes/'
for n in (37,38,16,28,35,36):
    e=f'{root}recon_{n:06d}';m=json.load(open(e+'/recon_meta.json'));mv=max(m['movers'],key=lambda x:x['move_cm']);s=mv['side']
    h=np.load(e+'/bundle/result.npz');O=h['object_to_world_transform'].astype(float);sc=float(1/h['object_scale']);N=len(O);op=O[:,:3,3]*sc;Ro=R.from_matrix(O[:,:3,:3])
    ext=np.array(m['objects'][mv['index']]['extent_cm']);ax=np.eye(3)[np.argmax(ext)]
    D={x:100*np.linalg.norm(h[f'hand_{x}_wrist_to_world_transform'][:,:3,3]*sc-op,axis=1) for x in('left','right')}
    # frame-to-frame body-frame rotation split into twist about the long axis and swing of the axis
    dR=(Ro[:-1].inv()*Ro[1:]).as_rotvec();tw=np.abs(dR@ax);sw=np.linalg.norm(dR-np.outer(dR@ax,ax),axis=1)
    axw=Ro.apply(ax);tot=np.degrees(np.arccos(np.clip(axw@axw[0],-1,1)))
    f=np.degrees(np.linalg.norm(dR,axis=1))
    near=np.where(D['left']<D['right'],0,1)
    on=mv['onset']
    hold=[(D[x][on:]<25).mean().round(2) for x in('left','right')]
    print(f"ep{n} ext {ext} axis {np.argmax(ext)} N={N}: path deg twist {np.degrees(tw.sum()):.0f} swing {np.degrees(sw.sum()):.0f}; long axis max tilt {tot.max():.0f}; frame jumps >15deg {(f>15).sum()} >30deg {(f>30).sum()}  pos jumps>3cm {(100*np.linalg.norm(np.diff(op,axis=0),axis=1)>3).sum()}; frac near(<25cm) after onset L/R {hold}")
    idx=list(range(0,N,max(1,N//14)))
    print('   frame',' '.join(f'{i:4d}' for i in idx));print('   L cm ',' '.join(f'{D["left"][i]:4.0f}' for i in idx));print('   R cm ',' '.join(f'{D["right"][i]:4.0f}' for i in idx));print('   axis ',' '.join(f'{tot[i]:4.0f}' for i in idx))
