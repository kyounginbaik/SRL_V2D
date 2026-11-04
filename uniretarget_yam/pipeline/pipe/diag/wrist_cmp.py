import sys,numpy as np,json,os
sys.path.insert(0,'/uniretarget_yam/code')
import recon_holds as H,trimesh
from pathlib import Path
from scipy.spatial.transform import Rotation as R
Z=Path('/scratch2/exaFLOPs/latest_recon/v2/best_reconstruction_20261103')
def cam_D(T,mesh,Hw,W):
    # Rigid map from the hand run's world to the target world, from the same camera seen in both (median over frames).
    C=Hw['camera_to_world_transform'].astype(float);n=min(len(C),len(W));sc=2/(float(Hw['hand_left_scale'].mean())+float(Hw['hand_right_scale'].mean()))
    A=C[:n,:3,3]*sc;B=W[:n,:3,3];Rs=[W[i,:3,:3]@C[i,:3,:3].T for i in range(n)];Rm=R.from_matrix(np.array(Rs)).mean();t=np.median(B-Rm.apply(A),axis=0)
    res=np.median(np.linalg.norm(Rm.apply(A)+t-B,axis=1))*100
    v=np.asarray(trimesh.load(mesh,force='mesh').vertices);v=v[::max(1,len(v)//300)];D=[]
    Ro=R.from_matrix(T[:n,:3,:3])
    for sd in('left','right'):
        w=Rm.apply(Hw[f'hand_{sd}_wrist_to_world_transform'][:n,:3,3].astype(float)*sc)+t;D.append(np.array([np.linalg.norm(Ro[f].apply(v)+T[f,:3,3]-w[f],axis=1).min() for f in range(n)])*100)
    return np.array(D),res
for ep in sys.argv[1:] or ['25','37','8','10','20','28','6','16','38']:
    n=int(ep);e=f'episode_{n:06d}';tg=Z/'targets'/e;hd=Z/'hands'/e;b=np.load(Z/'bundles'/e/'object_poses.npz');names=[str(x) for x in b['names']];W=b['world_from_rgb'];up=np.array(json.load(open(f'/uniretarget_yam/episodes/recon_{n:06d}/recon_meta.json'))['up_in_target_world'])
    for x in names:
        T=np.load(tg/f'{x}_trajectory.npz')['T_world_obj'];Hw=dict(np.load(hd/f'hawor_{x}_result.npz'));old=H.wrist_surface_cm(T,tg/f'{x}_mesh_metric.obj',Hw,up);new,res=cam_D(T,tg/f'{x}_mesh_metric.obj',Hw,W);s=max(1,len(T)//10)
        print(n,x[:14],'camfit',round(res,1),'cm');print('  old L',[int(a) for a in old[0,::s]],'R',[int(a) for a in old[1,::s]]);print('  new L',[int(a) for a in new[0,::s]],'R',[int(a) for a in new[1,::s]],flush=True)
