"""Make K32_G (interpolate only confirmed glitch poses) and K32_RP (temporal solve p>=35) from the K32_R10 bundle; spoon only, cup and p<35 unchanged. GT-free."""
import json,shutil,hashlib,numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as Rot,Slerp
K=Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK32_ep32');SRC=K/'bundles/K32_R10/episode_000032'
z=np.load(SRC/'object_poses.npz',allow_pickle=False);d0={k:z[k] for k in z.files};ks=list(d0['names']).index('wooden_spoon');n=len(d0['world_from_object'])
C0=d0['camera_from_object'][:,ks];WR=d0['world_from_rgb']
def interp(T,a,b):
    O=T.copy();s=Slerp([a,b],Rot.from_matrix(T[[a,b],:3,:3]))
    for p in range(a+1,b):
        w=(p-a)/(b-a);O[p,:3,:3]=s(p).as_matrix();O[p,:3,3]=(1-w)*T[a,:3,3]+w*T[b,:3,3]
    return O
G=interp(interp(C0,88,91),150,154)
RP=np.load(K/'recon_perfect/full_0.1.npz')['T'];assert np.allclose(RP[:35],C0[:35])
V14T=Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/target_final_v14_g17/episode_000032')
for lab,Cm in (('K32_G',G),('K32_RP',RP)):
    out=K/'bundles'/lab/'episode_000032';assert not out.exists();shutil.copytree(SRC,out)
    d={k:v.copy() for k,v in d0.items()};Wm=np.stack([d0['world_from_object'][t,ks] if np.array_equal(Cm[t],C0[t]) else WR[t]@Cm[t] for t in range(n)]);d['camera_from_object'][:,ks]=Cm;d['world_from_object'][:,ks]=Wm
    assert np.allclose(np.linalg.inv(WR[40])@d0['world_from_object'][40,ks],C0[40],atol=1e-5)
    (out/'object_poses.npz').unlink();np.savez_compressed(out/'object_poses.npz',**d)
    (out/'K32RP_CHANGES.json').write_text(json.dumps(dict(label=lab,source=str(SRC),ground_truth_used=False,sha=hashlib.sha256((out/'object_poses.npz').read_bytes()).hexdigest())))
    tg=K/'targets'/lab/'episode_000032';tg.mkdir(parents=True);np.savez_compressed(tg/'wooden_spoon_trajectory.npz',T_world_obj=Wm)
    for fn in ('blue_cup_trajectory.npz','blue_cup_mesh_metric.obj'):shutil.copy2(V14T/fn,tg/fn)
    shutil.copy2(out/'meshes'/'wooden_spoon.obj',tg/'wooden_spoon_mesh_metric.obj')
    W0=d0['world_from_object'][:,ks];ch=np.degrees((Rot.from_matrix(W0[:,:3,:3]).inv()*Rot.from_matrix(Wm[:,:3,:3])).magnitude());print(lab,'frames changed',int((ch>1e-3).sum()),'max deg %.1f'%ch.max(),'first',int(np.argmax(ch>1e-3)))
