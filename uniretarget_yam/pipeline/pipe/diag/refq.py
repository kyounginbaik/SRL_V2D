import sys,json;sys.path.insert(0,'/uniretarget_yam/code')
import numpy as np,prepare_recon as PR,recon_holds as H
from pathlib import Path
from scipy.spatial.transform import Rotation as R
Z=Path('/z');show=[20]
for d in sorted((Z/'targets').iterdir()):
    ep=d.name;n=int(ep[-6:]);bd=Z/'bundles'/ep;hd=Z/'hands'/ep;zp=np.load(bd/'object_poses.npz');names=[str(x) for x in zp['names']];f=np.load(hd/f'hawor_{names[0]}_result.npz');up,Rw=PR.up_and_heading(zp,f);T=np.stack([np.load(d/f'{x}_trajectory.npz')['T_world_obj'] for x in names],1);s=H.schedule(d,hd,names,T,up);row=[]
    for h in s['holds']:
        k=h['index'];o=names[k];Hd=dict(np.load(hd/f'hawor_{o}_result.npz'));D=H.wrist_surface_cm(T[:,k],d/f'{o}_mesh_metric.obj',Hd,up)[0 if h['side']=='left' else 1];a,b=h['start'],h['end'];seg=D[a:b+1];off=seg>18
        sc=2/(float(Hd['hand_left_scale'].mean())+float(Hd['hand_right_scale'].mean()));om=Hd['object_to_world_transform'][:,:3,3].astype(float)*sc;tp=T[:,k,:3,3];fit=R.align_vectors(om-om.mean(0),tp-tp.mean(0))[0];res=100*np.linalg.norm(fit.apply(tp-tp.mean(0))-(om-om.mean(0)),axis=1)
        row.append(f"{o[:10]}:{h['side'][0].upper()}[{a}-{b}] off>18cm {off.mean():.2f} median {np.median(seg):.0f}cm max {seg.max():.0f}cm objfit {res.mean():.1f}/{res.max():.1f}cm")
    print(n,' ; '.join(row))
    if n in show:
        for k,o in enumerate(names):
            Hd=dict(np.load(hd/f'hawor_{o}_result.npz'));D=H.wrist_surface_cm(T[:,k],d/f'{o}_mesh_metric.obj',Hd,up);mv=100*np.linalg.norm(T[:,k,:3,3]-T[0,k,:3,3],axis=1);sc=2/(float(Hd['hand_left_scale'].mean())+float(Hd['hand_right_scale'].mean()));W=Hd['hand_right_wrist_to_world_transform'][:,:3,3].astype(float)*sc;sp=100*np.r_[0,np.linalg.norm(np.diff(W,axis=0),axis=1)]*20;idx=list(range(0,len(mv),12))
            print('  ',o,'frame ',' '.join(f'{i:4d}' for i in idx));print('    obj move cm',' '.join(f'{mv[i]:4.0f}' for i in idx));print('    L wrist-surf',' '.join(f'{D[0,i]:4.0f}' for i in idx));print('    R wrist-surf',' '.join(f'{D[1,i]:4.0f}' for i in idx));print('    R speed cm/s',' '.join(f'{sp[i]:4.0f}' for i in idx));print('    valid L/R',Hd['hand_left_is_valid'].mean().round(2),Hd['hand_right_is_valid'].mean().round(2),[k_ for k_ in Hd if 'conf' in k_ or 'valid' in k_])
