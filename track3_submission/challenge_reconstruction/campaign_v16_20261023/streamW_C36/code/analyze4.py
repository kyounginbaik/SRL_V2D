import sys, numpy as np
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamW_C36/code'); import perframe as P
from scipy.spatial.transform import Rotation as R
orig = np.load('/data1/SRL_V2D/challenge_reconstruction/final_recon/episode_000036/object_poses.npz')['world_from_object'][:, 0]
Tc = P.TC @ P.Gc(); A0 = orig[0] @ np.linalg.inv(Tc[0]); Ref = A0 @ Tc
D = np.einsum('nji,njk->nik', orig[:, :3, :3], Ref[:, :3, :3]); rv = R.from_matrix(D).as_rotvec()
for th in [20, 30, 45, 60, 90, 120, 180]:
    m = np.linalg.norm(rv, axis=1); s = np.minimum(1, np.radians(th) / np.maximum(m, 1e-9))
    T = orig.copy(); T[:, :3, :3] = orig[:, :3, :3] @ R.from_rotvec(rv * s[:, None]).as_matrix(); T[:, :3, 3] = Ref[:, :3, 3]
    p, o = P.errors(T); print(f'clip {th:3d}: {p.mean():.2f} cm {o.mean():.1f} deg')
p, o = P.errors(orig); print('orig', p.mean(), o.mean())
T = orig.copy(); T[:, :3, 3] = Ref[:, :3, 3]; p, o = P.errors(T); print('orig rot + ref pos', p.mean(), o.mean())
