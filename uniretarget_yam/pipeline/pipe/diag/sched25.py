import sys,numpy as np,json
sys.path.insert(0,'/uniretarget_yam/code')
import recon_holds as H
from pathlib import Path
from scipy.spatial.transform import Rotation as R
Z=Path('/scratch2/exaFLOPs/latest_recon/v2/best_reconstruction_20261103');n='episode_000025';tg=Z/'targets'/n;hd=Z/'hands'/n;names=['wooden_piece_1','wooden_piece_2']
T=np.stack([np.load(tg/f'{x}_trajectory.npz')['T_world_obj'] for x in names],1)
up=np.array(json.load(open('/uniretarget_yam/episodes/recon_000025/recon_meta.json'))['up_in_target_world'])
print('MAX_STAGES',H.MAX_STAGES,'NEAR',H.NEAR_CM,'COHOLD',H.COHOLD_MIN,'MIN_LEN',H.MIN_LEN)
for k,x in enumerate(names):
    D=H.wrist_surface_cm(T[:,k],tg/f'{x}_mesh_metric.obj',dict(np.load(hd/f'hawor_{x}_result.npz')),up)
    print(x,'left',[int(D[0,f]) for f in range(0,len(T),30)]);print(x,'right',[int(D[1,f]) for f in range(0,len(T),30)])
    print(x,'pos move',[int(100*np.linalg.norm(T[f,k,:3,3]-T[0,k,:3,3])) for f in range(0,len(T),30)])
s=H.schedule(tg,hd,names,T,up);print(json.dumps({k:v for k,v in s.items() if k in('holds','frozen','primary','kinds','end')})[:900])
