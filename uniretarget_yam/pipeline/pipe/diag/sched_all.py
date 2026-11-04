import sys,numpy as np,json,os
sys.path.insert(0,'/uniretarget_yam/code')
import recon_holds as H
from pathlib import Path
Z=Path('/scratch2/exaFLOPs/hawor_rerun/root')
for e in sorted(os.listdir(Z/'bundles')):
    n=int(e[-6:]);b=np.load(Z/'bundles'/e/'object_poses.npz');names=[str(x) for x in b['names']];tg=Z/'targets'/e;hd=Z/'hands'/e
    T=np.stack([np.load(tg/f'{x}_trajectory.npz')['T_world_obj'] for x in names],1)
    mp=f'/uniretarget_yam/episodes/recon_{n:06d}/recon_meta.json';up=np.load(tg.parent.parent/'hands'/e/f'hawor_{names[0]}_result.npz')
    if os.path.exists(mp):u=np.array(json.load(open(mp))['up_in_target_world'])
    else:
        import prepare_recon as PR
        u,_=PR.up_and_heading(np.load(Z/'bundles'/e/'object_poses.npz'),up)
    s=H.schedule(tg,hd,names,T,u);mv=[int(100*np.linalg.norm(T[:,k,:3,3]-T[0,k,:3,3],axis=1).max()) for k in range(len(names))]
    print(n,[f'{x[:10]} {m}cm' for x,m in zip(names,mv)],s.get('kinds'),[f"{h['object'][:6]+h['object'][-2:]}:{h['side'][0].upper()}[{h['start']}-{h['end']}]" for h in s['holds']],'frozen',[names[k][:10] for k in s['frozen']],'end',s['end'],'of',len(T),flush=True)
