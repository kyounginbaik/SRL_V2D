import sys,json
sys.path.insert(0,'.')
import numpy as np,torch
from scipy.spatial.transform import Rotation as R
from hawor.utils.process import run_mano,run_mano_left
Z='/scratch2/exaFLOPs/latest_recon/v2/best_reconstruction_20261103/';n=25;e=f'episode_{n:06d}'
d=np.load(f'../root/hands/{e}/hawor_wooden_piece_2_result.npz');C=d['camera_to_world_transform'].astype(float);b=np.load(Z+f'bundles/{e}/object_poses.npz');W=b['world_from_rgb'];O=b['world_from_object'];Si=W[0]@np.linalg.inv(C[0])
TAG=sys.argv[1];print('=====',TAG);D=np.load(f'/scratch2/exaFLOPs/SRL_V2D/uniretarget_yam/episodes/recon_{n:06d}/yam/flash/mink_debug_{TAG}/ik_debug.npz');names=[str(x) for x in D['names']];op=D['pos_object'];oq=D['quat_object'];k=1;F=len(W)
Sim=np.eye(4);Sim[:3,:3]=R.from_quat(oq[k][0]).as_matrix();Sim[:3,3]=op[0,k];Sk=Sim@np.linalg.inv(O[0,k])
tt=lambda x:torch.as_tensor(np.asarray(x),dtype=torch.float32).cuda()[None]
TIP={'thumb':4,'index':8,'middle':12,'ring':16,'pinky':20}
for s in('left','right'):
    X=d[f'hand_{s}_wrist_to_world_transform'].astype(float);fn=run_mano if s=='right' else run_mano_left;out=fn(tt(X[:,:3,3]),tt(R.from_matrix(X[:,:3,:3]).as_rotvec()),tt((d[f'hand_{s}_finger_pose']*(np.array([1,-1,-1.]) if s=='left' else 1)).reshape(F,45)),betas=tt(np.tile(d[f'hand_{s}_betas'],(F,1))));J=out['joints'][0].cpu().numpy().astype(float);V=out['vertices'][0].cpu().numpy().astype(float)
    def sim(P):
        q=(Si[:3,:3]@P.reshape(-1,3).T).T+Si[:3,3];q=(Sk[:3,:3]@q.T).T+Sk[:3,3];return q.reshape(P.shape)
    Vs=sim(V);Js=sim(J);i=[names.index(f'{s}_{x}_fingertip') for x in TIP];tips=np.stack([Js[:,TIP[x]] for x in TIP],1)
    for lab,arr in (('raw',D['raw']),('goal',D['goal']),('robot',D['robot'])):
        a=arr[:,i];e0=np.linalg.norm(a-tips,axis=2)*100;cen=np.linalg.norm(a.mean(1)-tips.mean(1),axis=1)*100;print(s,lab,'tips vs HaWoR tips cm mean %.1f, hand-centre offset mean %.1f p95 %.1f'%(e0.mean(),cen.mean(),np.percentile(cen,95)))
    p=names.index(f'{s}_hand_C_MC');im=names.index(f'{s}_index_MP');rm=names.index(f'{s}_ring_MP')
    def nrm(A):
        n_=np.cross(A[:,im]-A[:,p],A[:,rm]-A[:,p]);return n_/np.linalg.norm(n_,axis=1,keepdims=True)
    def ang(u,v):return np.degrees(np.arccos(np.clip((u*v).sum(1),-1,1)))
    hw=(Js[:,0]);nm=lambda:None
    # MANO palm normal from wrist, index base and ring base joints
    idx,rng=[5,13]
    nh=np.cross(Js[:,idx]-Js[:,0],Js[:,rng]-Js[:,0]);nh/=np.linalg.norm(nh,axis=1,keepdims=True)
    print(s,'palm normal angle deg: raw vs HaWoR %.0f | goal vs raw %.0f | robot vs goal %.0f (p95 %.0f) | robot vs HaWoR %.0f (p95 %.0f)'%(ang(nrm(D['raw']),nh).mean(),ang(nrm(D['goal']),nrm(D['raw'])).mean(),ang(nrm(D['robot']),nrm(D['goal'])).mean(),np.percentile(ang(nrm(D['robot']),nrm(D['goal'])),95),ang(nrm(D['robot']),nh).mean(),np.percentile(ang(nrm(D['robot']),nh),95)))
    print(s,'finger spread cm (tip to tip extent) HaWoR %.1f raw %.1f robot %.1f'%(np.linalg.norm(tips-tips.mean(1,keepdims=True),axis=2).mean()*100,np.linalg.norm(D['raw'][:,i]-D['raw'][:,i].mean(1,keepdims=True),axis=2).mean()*100,np.linalg.norm(D['robot'][:,i]-D['robot'][:,i].mean(1,keepdims=True),axis=2).mean()*100))
