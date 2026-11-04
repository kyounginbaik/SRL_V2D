"""Bundle + scorer target K33_<LABEL> from K33_H5r16 + rr_solve solution: spoon pose replaced on the solved frames, cup and everything else untouched, same mesh."""
import sys, json, shutil, hashlib, numpy as np
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent)); from rr_apply import apply
K=Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK33_ep33'); lab,scan,sol=sys.argv[1],sys.argv[2],sys.argv[3]
SRC=K/'bundles/K33_H5r16/episode_000033'; out=K/'bundles'/lab/'episode_000033'; assert not out.exists(); shutil.copytree(SRC,out)
z=np.load(SRC/'object_poses.npz',allow_pickle=False); d={k:z[k] for k in z.files}; sc=np.load(scan); so=np.load(sol); C,W,k=apply(z,sc,so)
assert np.allclose(W[:60],d['world_from_object'][:60,k]); d['camera_from_object'][:,k]=C; d['world_from_object'][:,k]=W
(out/'object_poses.npz').unlink(); np.savez_compressed(out/'object_poses.npz',**d)
(out/'K33_CHANGES.json').write_text(json.dumps(dict(stream='RR33',label=lab,source=str(SRC),ground_truth_used=False,scan=scan,solution=sol,roll_deg=[round(float(x),2) for x in so['roll']],tilt_deg=np.round(so['tilt'],2).tolist(),shift_px=np.round(so['shift'],2).tolist(),object_poses_sha256=hashlib.sha256((out/'object_poses.npz').read_bytes()).hexdigest()),indent=0))
tg=K/'targets'/lab/'episode_000033'; tg.mkdir(parents=True); ref=K/'targets/K33_H5r16/episode_000033'
for fn in ('blue_cup_trajectory.npz','blue_cup_mesh_metric.obj','wooden_spoon_mesh_metric.obj'): shutil.copy2(ref/fn,tg/fn)
np.savez_compressed(tg/'wooden_spoon_trajectory.npz',T_world_obj=W)
from scipy.spatial.transform import Rotation as Rot
R=Rot.from_matrix(W[:,:3,:3]); w=np.degrees((R[1:]*R[:-1].inv()).magnitude()); R0=Rot.from_matrix(z['world_from_object'][:,k,:3,:3]); w0=np.degrees((R0[1:]*R0[:-1].inv()).magnitude()); ch=np.degrees((R*R0.inv()).magnitude()); dp=np.linalg.norm(W[:,:3,3]-z['world_from_object'][:,k,:3,3],axis=1)*100
print(lab,'max deg/frame',round(w.max(),1),'base',round(w0.max(),1),'frames>30',(w>30).sum()); print(' change deg per s: '+' '.join(f'{s//20}s:{ch[s:s+20].mean():.0f}' for s in range(0,len(ch),20))); print(' change cm per s: '+' '.join(f'{s//20}s:{dp[s:s+20].mean():.1f}' for s in range(0,len(dp),20)))
