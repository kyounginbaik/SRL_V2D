"""Scan current eval final tracks for half-turn twin flips (frame-to-frame jumps a principal-axis twin explains) and per-frame jumps."""
import sys, json, numpy as np, glob, os, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'/data1/SRL_V2D/challenge_reconstruction/public_gt_check_20261020/code/postprocess'); import pp
from scipy.spatial.transform import Rotation as R
SRC={**{e:f'/data1/SRL_V2D/challenge_reconstruction/final_recon/episode_{e:06d}' for e in (1,3,5,6,8,10,19,32,33,35,36,37,38)},
     **{e:f'/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/physics_collision_v2/input_root/challenge_reconstruction/final_recon/episode_{e:06d}' for e in (14,15,16,20,25,28,30)}}
rows=[]
for e,d in sorted(SRC.items()):
    z=np.load(d+'/object_poses.npz'); names=[str(x) for x in z['names']]; T=z['world_from_object']
    for i,n in enumerate(names):
        mp=f'{d}/assets/{n}_visual.obj'
        if not os.path.exists(mp): mp=f'{d}/assets/{n}.obj'
        tw,c,w=pp.principal_twins(mp); Ti=T[:,i]; _,picks=pp.continuity(Ti,tw,c)
        jumps=np.array([pp.rot_dist(Ti[t-1][:3,:3],Ti[t][:3,:3]) for t in range(1,len(Ti))])
        flips=np.where(np.diff(np.r_[0,picks])!=0)[0]
        rows.append(dict(ep=e,obj=n,frames=len(Ti),twin_switch_frames=[int(x) for x in flips][:12],twin_frames=int((picks!=0).sum()),max_jump_deg=float(jumps.max()),jumps_over_20=int((jumps>20).sum())))
        print(f"ep{e:2d} {n:18s} twin-held frames {rows[-1]['twin_frames']:4d}/{len(Ti)} switches at {rows[-1]['twin_switch_frames'][:6]} | max frame jump {jumps.max():5.1f} deg, >20deg jumps {rows[-1]['jumps_over_20']}")
json.dump(rows,open('/data1/SRL_V2D/challenge_reconstruction/recon_v14_20261022/flip_scan_current.json','w'),indent=1)
