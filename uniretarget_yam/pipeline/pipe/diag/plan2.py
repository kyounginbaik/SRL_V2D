import sys;sys.path.insert(0,'/uniretarget_yam/code')
import numpy as np,prepare_recon as PR,recon_holds as H
from pathlib import Path
Z=Path('/z')
for d in sorted((Z/'targets').iterdir()):
    ep=d.name;bd=Z/'bundles'/ep;hd=Z/'hands'/ep;zp=np.load(bd/'object_poses.npz');names=[str(x) for x in zp['names']];f=np.load(hd/f'hawor_{names[0]}_result.npz');up,Rw=PR.up_and_heading(zp,f);T=np.stack([np.load(d/f'{n}_trajectory.npz')['T_world_obj'] for n in names],1);a,h,r=H.plan_all(d,hd,names,T,up)
    print(int(ep[-6:]),'cover',a['coverage'],' + '.join(f"{x['object']}:{x['side']}[{x['onset']}-{x['end']}]" for x in h),'| stays:',[names[k] for k in r])
