"""Ep38: compare the dust pan mesh at the reference pose and at the robot's achieved pose against stereo depth inside the
object mask. For each frame: back-project masked depth pixels to world, nearest-surface residual to the posed mesh;
report median distance and median world-z residual (measured minus mesh). Also the scan mesh (scale check)."""
import glob, json, numpy as np, cv2, trimesh, pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
F='/data1/SRL_V2D/challenge_reconstruction/final_recon/episode_000038'; DEP='/data1/SRL_V2D/challenge_reconstruction/eval_stereo_current_20261010/episode_000038/depth_rgb'
MS='/data1/SRL_V2D/challenge_submissions/track3/eval_hawor_latest_20261008/outputs/episode_000038/mini_dust_pan/masks'
P='/data1/SRL_V2D/challenge_submissions/track3_v9_20261021/all_20_episodes'; OUT='/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep38/diag'
z=np.load(F+'/object_poses.npz'); Wo=z['world_from_object'][:,0]; Wc=z['world_from_rgb']; K=np.load(F+'/camera_motion.npz')['K']
mesh=trimesh.load(P+'/meshes/episode_000038/mini_dust_pan.obj',force='mesh'); S=trimesh.sample.sample_surface(mesh,20000,seed=0)[0]
d=pd.read_parquet(P+'/results/episode_000038.parquet').sort_values('step')
A=np.tile(np.eye(4),(len(d),1,1)); A[:,:3,:3]=R.from_quat(d[[f'achieved_quaternion_{a}' for a in 'xyzw']].to_numpy()).as_matrix(); A[:,:3,3]=d[[f'achieved_position_{a}' for a in 'xyz']].to_numpy()
masks={k:sorted(glob.glob(f'{MS}/{k}/*.png')) for k in ('1','2','3')}
rows=[]
for k in range(0,263,5):
    enc=cv2.imread(f'{DEP}/{k:06d}.png',-1).astype(float); valid=enc<65535; dep=np.where(valid,65535./np.maximum(enc,1)-1,0)
    m=cv2.resize(cv2.imread(masks['1'][k],0),(dep.shape[1],dep.shape[0]),interpolation=cv2.INTER_NEAREST)>0
    m=cv2.erode(m.astype(np.uint8),np.ones((5,5),np.uint8))>0
    for h in ('2','3'):
        hm=cv2.resize(cv2.imread(masks[h][k],0),(dep.shape[1],dep.shape[0]),interpolation=cv2.INTER_NEAREST)>0; m&=~cv2.dilate(hm.astype(np.uint8),np.ones((9,9),np.uint8)).astype(bool)
    v,u=np.nonzero(m&valid&(dep>0.15)&(dep<1.5))
    if len(u)<200: rows.append(dict(t=k/20,n=int(len(u)))); continue
    zc=dep[v,u]; X=np.stack([(u-K[0,2])*zc/K[0,0],(v-K[1,2])*zc/K[1,1],zc],1); Xw=X@Wc[k,:3,:3].T+Wc[k,:3,3]
    r=dict(t=k/20,n=int(len(u)))
    for tag,T in (('ref',Wo[k]),('robot',A[min(k,len(A)-1)])):
        Mw=S@T[:3,:3].T+T[:3,3]; dist,idx=cKDTree(Mw).query(Xw); res=Xw-Mw[idx]
        r[tag+'_dist_cm']=float(np.median(dist)*100); r[tag+'_dz_cm']=float(np.median(res[:,2])*100)
        r[tag+'_centroid_dz_cm']=float((Xw[:,2].mean()-Mw[:,2].mean())*100)  # rough
    rows.append(r)
json.dump(rows,open(OUT+'/depth_check.json','w'),indent=1)
for r in rows:
    if 'ref_dist_cm' in r: print(f"t={r['t']:5.2f} n={r['n']:5d} | ref: med dist {r['ref_dist_cm']:.2f} cm, med dz {r['ref_dz_cm']:+.2f} | robot: med dist {r['robot_dist_cm']:.2f}, med dz {r['robot_dz_cm']:+.2f}")
    else: print(f"t={r['t']:5.2f} n={r['n']} (too few valid masked depth pixels)")
