import sys,json;sys.path.insert(0,'/uniretarget_yam/code')
import numpy as np,prepare_recon as PR,recon_holds as H
from pathlib import Path
from scipy.spatial.transform import Rotation as R
Z=Path('/z')
for d in sorted((Z/'targets').iterdir()):
    ep=d.name;bd=Z/'bundles'/ep;hd=Z/'hands'/ep;zp=np.load(bd/'object_poses.npz');names=[str(x) for x in zp['names']];f=np.load(hd/f'hawor_{names[0]}_result.npz');up,Rw=PR.up_and_heading(zp,f);T=np.stack([np.load(d/f'{n}_trajectory.npz')['T_world_obj'] for n in names],1)
    h=H.plan(d,hd,names,T,up);F=h['end']+1;X=T[:F,h['mover']];p,r,c=H.condition(X[:,:3,3],R.from_matrix(X[:,:3,:3]));rise=100*np.linalg.norm(X[:,:3,3]-X[0,:3,3],axis=1).max();tr=np.degrees((R.from_matrix(X[:,:3,:3])*R.from_matrix(X[0,:3,:3]).inv()).magnitude()).max()
    print(f"{int(ep[-6:]):3d} {h['object'][:14]:14s} {h['side']:5s} onset {h['onset']:3d} end {h['end']:3d}/{h['full_frames']-1:3d} cover {h['coverage']:.2f} handoff {h['handoff']} move {rise:3.0f}cm turn {tr:3.0f}d smooth {c['sigma']} dev {c['dev_deg_mean']}/{c['dev_deg_max']}d {c['dev_cm_mean']}/{c['dev_cm_max']}cm glitch {c['glitch_frames']} K={len(names)}")
