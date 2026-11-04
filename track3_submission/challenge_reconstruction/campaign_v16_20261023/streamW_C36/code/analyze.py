import sys, numpy as np, pandas as pd
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamW_C36/code'); import perframe as P
from scipy.spatial.transform import Rotation as R
def run_T(p, col='achieved'):
    r = pd.read_parquet(p); r = r[r.environment_id == 0].sort_values('reference_frame')
    T = np.tile(np.eye(4), (len(r), 1, 1)); T[:, :3, 3] = r[[f'{col}_position_{a}' for a in 'xyz']].values
    T[:, :3, :3] = R.from_quat(r[[f'{col}_quaternion_{a}' for a in 'xyzw']].values).as_matrix(); return T
src = {'orig': np.load('/data1/SRL_V2D/challenge_reconstruction/final_recon/episode_000036/object_poses.npz')['world_from_object'][:, 0],
       'corr': np.load('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep36/source_corrected/episode_000036/object_poses.npz')['world_from_object'][:, 0]}
G = '/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates/G_ep36_margins/episode_000036/results/episode_000036.parquet'
src['v21_ref'] = run_T(G, 'reference'); src['v21_ach'] = run_T(G)
for k, T in src.items():
    p, o = P.errors(T); print(f'{k:8s} mean {p.mean():.2f} cm {o.mean():.1f} deg')
    for a in range(0, 407, 20): print(f'   f{a:3d}-{a+19:3d}  {p[a:a+20].mean():5.2f} {o[a:a+20].mean():6.1f}  max {o[a:a+20].max():6.1f}')
