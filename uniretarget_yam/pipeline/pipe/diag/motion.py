import numpy as np,json,sys
from scipy.spatial.transform import Rotation as R
root='/uniretarget_yam/episodes/'
for n in (37,38,16,28,35,36):
    e=f'{root}recon_{n:06d}';m=json.load(open(e+'/recon_meta.json'));mv=max(m['movers'],key=lambda x:x['move_cm']);s=mv['side']
    h=np.load(e+'/bundle/result.npz');O=h['object_to_world_transform'].astype(float);W=h[f'hand_{s}_wrist_to_world_transform'].astype(float)
    sc=float(1/h['object_scale']);N=len(O);op=O[:,:3,3]*sc;wp=W[:,:3,3]*sc;Ro=R.from_matrix(O[:,:3,:3]);Rw=R.from_matrix(W[:,:3,:3])
    rel=Rw.inv()*Ro;rp=Rw.inv().apply(op-wp);on=mv['onset'];ref=min(on+10,N-1)
    ang=np.degrees((rel*rel[ref].inv()).magnitude());dp=100*np.linalg.norm(rp-rp[ref],axis=1)
    w=np.degrees((Ro[1:]*Ro[:-1].inv()).magnitude())*20;v=100*np.linalg.norm(np.diff(op,axis=0),axis=1)*20
    turn=np.degrees((Ro*Ro[0].inv()).magnitude());d0=100*np.linalg.norm(op-op[0],axis=1);dist=100*np.linalg.norm(wp-op,axis=1)
    print(f"ep{n} {m['objects'][mv['index']]['name']} {s} N={N} onset={on}  obj speed cm/s p50 {np.median(v):.0f} max {v.max():.0f} | ang speed deg/s p50 {np.median(w):.0f} p99 {np.percentile(w,99):.0f} max {w.max():.0f} at {w.argmax()} | n>200deg/s {(w>200).sum()}")
    idx=list(range(0,N,max(1,N//14)))
    print('  frame      ',' '.join(f'{i:5d}' for i in idx))
    print('  rise_cm    ',' '.join(f'{d0[i]:5.0f}' for i in idx))
    print('  turn_deg   ',' '.join(f'{turn[i]:5.0f}' for i in idx))
    print('  wrist-obj  ',' '.join(f'{dist[i]:5.0f}' for i in idx))
    print('  relRot_deg ',' '.join(f'{ang[i]:5.0f}' for i in idx))
    print('  relPos_cm  ',' '.join(f'{dp[i]:5.0f}' for i in idx))
