"""Assemble K32_RP2 = K32_R10 bundle with spoon := mergedw_spoon (marks x4 merge, f0-20 kept) and cup := cup_lam0.4_sg_rot9_pos7 (f0-5 kept). GT-free."""
import json,shutil,hashlib,numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as Rot
K=Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK32_ep32');SRC=K/'bundles/K32_R10/episode_000032';RP=K/'recon_perfect'
z=np.load(SRC/'object_poses.npz',allow_pickle=False);d={k:z[k].copy() for k in z.files};names=list(d['names']);WR=d['world_from_rgb'];n=len(WR)
new={'wooden_spoon':np.load(RP/'mergedw_spoon.npy'),'blue_cup':np.load(RP/'cup_lam0.4_sg_rot9_pos7.npy')};keep={'wooden_spoon':21,'blue_cup':6}
lab='K32_RP2';out=K/'bundles'/lab/'episode_000032';assert not out.exists();shutil.copytree(SRC,out);rec={}
for nm,Cm in new.items():
    ks=names.index(nm);C0=d['camera_from_object'][:,ks];assert np.allclose(Cm[:keep[nm]],C0[:keep[nm]],atol=1e-9),nm
    Wm=np.stack([d['world_from_object'][t,ks] if np.allclose(Cm[t],C0[t]) else WR[t]@Cm[t] for t in range(n)])
    ch=np.degrees((Rot.from_matrix(C0[:,:3,:3]).inv()*Rot.from_matrix(Cm[:,:3,:3])).magnitude());dt=100*np.linalg.norm(Cm[:,:3,3]-C0[:,:3,3],axis=1)
    d['camera_from_object'][:,ks]=Cm;d['world_from_object'][:,ks]=Wm;rec[nm]=dict(frames_changed=int((ch>1e-6).sum()),first_changed=int(np.argmax(ch>1e-6)),rot_deg_mean=float(ch.mean()),rot_deg_max=float(ch.max()),cm_mean=float(dt.mean()),cm_max=float(dt.max()))
    tg=K/'targets'/lab/'episode_000032';tg.mkdir(parents=True,exist_ok=True);np.savez_compressed(tg/f'{nm}_trajectory.npz',T_world_obj=Wm);shutil.copy2(out/'meshes'/f'{nm}.obj',tg/f'{nm}_mesh_metric.obj')
(out/'object_poses.npz').unlink();np.savez_compressed(out/'object_poses.npz',**d)
rec.update(label=lab,source=str(SRC),ground_truth_used=False,method='recon_perfect merge.py (masks+stereo+spikes+user marks+temporal; spoon MW=4) + merge_cup2.py (cotracker tracks, LAM 0.4) + SG smooth rot9/pos7',sha=hashlib.sha256((out/'object_poses.npz').read_bytes()).hexdigest()[:16])
(out/'K32RP2_CHANGES.json').write_text(json.dumps(rec,indent=1));print(json.dumps(rec,indent=1))
# sanity: target file format matches K32_R10's
for f in ('wooden_spoon_trajectory.npz','blue_cup_trajectory.npz'):
    a=np.load(K/'targets/K32_R10/episode_000032'/f);b=np.load(K/'targets'/lab/'episode_000032'/f);assert a.files==b.files and a['T_world_obj'].shape==b['T_world_obj'].shape,f
print('targets ok',sorted(p.name for p in (K/'targets'/lab/'episode_000032').iterdir()))
