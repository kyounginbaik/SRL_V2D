import numpy as np,json,glob,os
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import gaussian_filter1d as G
B='/z/'
for n in (37,38,16,28,35,36):
    ep=f'episode_{n:06d}';z=np.load(B+'bundles/'+ep+'/object_poses.npz');names=[str(x) for x in z['names']];e=f'/uniretarget_yam/episodes/recon_{n:06d}';m=json.load(open(e+'/recon_meta.json'));mv=max(m['movers'],key=lambda x:x['move_cm']);o=mv['object']
    T=np.load(B+f'targets/{ep}/{o}_trajectory.npz')['T_world_obj'];H=np.load(B+f'hands/{ep}/hawor_{o}_result.npz');print(f'ep{n} target frames {len(T)} bundle {len(z["world_from_object"])} hawor {len(H["hand_right_wrist_to_world_transform"])} keys-fps',[k for k in H.files if 'fps' in k or 'time' in k],'valid L/R',H['hand_left_is_valid'].mean().round(2),H['hand_right_is_valid'].mean().round(2))
    Ro=R.from_matrix(T[:,:3,:3]);q=Ro.as_quat();
    for t in range(1,len(q)):
        if q[t]@q[t-1]<0:q[t]=-q[t]
    for fs in (1.5,4,8,12):
        qs=G(q,fs,axis=0,mode='nearest');Rs=R.from_quat(qs/np.linalg.norm(qs,axis=1,keepdims=True));w=np.degrees((Rs[1:]*Rs[:-1].inv()).magnitude())*20;dev=np.degrees((Rs*Ro.inv()).magnitude());ps=G(T[:,:3,3],fs,axis=0,mode='nearest');dp=100*np.linalg.norm(ps-T[:,:3,3],axis=1)
        print(f'   sigma {fs:4} frames: speed deg/s p50 {np.median(w):4.0f} p99 {np.percentile(w,99):4.0f} max {w.max():4.0f} >172deg/s(3rad/s) {(w>172).sum():3d} | deviation from raw deg mean {dev.mean():4.1f} p95 {np.percentile(dev,95):4.1f} max {dev.max():4.1f} | pos cm mean {dp.mean():.2f} max {dp.max():.2f}')
