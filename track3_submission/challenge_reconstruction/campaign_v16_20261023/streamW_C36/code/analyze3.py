import sys, numpy as np
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamW_C36/code'); import perframe as P
from scipy.spatial.transform import Rotation as R
corr = np.load('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep36/source_corrected/episode_000036/object_poses.npz')
orig = np.load('/data1/SRL_V2D/challenge_reconstruction/final_recon/episode_000036/object_poses.npz')['world_from_object'][:, 0]
Tw = corr['world_from_object'][:, 0]; Tc = P.TC @ P.Gc(); A0 = Tw[0] @ np.linalg.inv(Tc[0]); Ref = A0 @ Tc
for f in range(0, 407, 6):
    Rl = Tw[f,:3,:3].T @ Ref[f,:3,:3]   # body-frame delta
    # decompose: roll about body x then residual
    ex = R.from_matrix(Rl).as_euler('xyz', degrees=True)
    Ro = orig[f,:3,:3].T @ Ref[f,:3,:3]; eo = R.from_matrix(Ro).as_euler('xyz', degrees=True)
    print(f'f{f:3d} t{f/20:5.2f} corr->ref body euler xyz {np.round(ex,0)}  orig->ref {np.round(eo,0)}  dpos {np.linalg.norm(Ref[f,:3,3]-Tw[f,:3,3])*100:.1f} z {Tw[f,2,3]*100:.1f}/{Ref[f,2,3]*100:.1f}')
