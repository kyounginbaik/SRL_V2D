import sys;sys.path.insert(0,'/uniretarget_yam/code')
import numpy as np,prepare_recon as PR,recon_holds as H
from pathlib import Path
Z=Path('/z');c={}
for d in sorted((Z/'targets').iterdir()):
    ep=d.name;bd=Z/'bundles'/ep;hd=Z/'hands'/ep;zp=np.load(bd/'object_poses.npz');names=[str(x) for x in zp['names']];f=np.load(hd/f'hawor_{names[0]}_result.npz');up,Rw=PR.up_and_heading(zp,f);T=np.stack([np.load(d/f'{n}_trajectory.npz')['T_world_obj'] for n in names],1);s=H.schedule(d,hd,names,T,up)
    for k in s['kinds']:c[k]=c.get(k,0)+1
    print(int(ep[-6:]),'cover',s['coverage'],s['kinds'],' ; '.join(f"{x['object'][:9]}:{x['side'][0].upper()}[{x['start']}-{x['end']}{' rel '+str(x['release']) if x['release'] else ''}]" for x in s['holds']),'| stays',[names[k][:9] for k in s['frozen']])
print(c)
