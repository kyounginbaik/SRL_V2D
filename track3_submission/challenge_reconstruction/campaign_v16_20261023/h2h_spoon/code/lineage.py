"""Check which source each v20 component's reference matches: ref_setup[f] = A @ W_src[f] @ B (A, B constant).
Fit A from frame 0 assuming B = I (same mesh frame), report max position / rotation residual over frames."""
import numpy as np, pandas as pd, re, sys
from scipy.spatial.transform import Rotation as R
C = '/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/'
args = open(C + 'assemble_v20.args').read()
FR = '/data1/SRL_V2D/challenge_reconstruction/final_recon/episode_%06d/object_poses.npz'
FR7 = '/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/input_root/challenge_reconstruction/final_recon/episode_%06d/object_poses.npz'
def T(df):
    p = df[[f'reference_position_{a}' for a in 'xyz']].values; q = df[[f'reference_quaternion_{a}' for a in 'xyzw']].values
    M = np.tile(np.eye(4), (len(p), 1, 1)); M[:, :3, :3] = R.from_quat(q).as_matrix(); M[:, :3, 3] = p; return M
for ep in [3, 5, 6, 15, 16, 32, 33, 35, 36, 37]:
    comp = re.search(rf'--component\s+{ep}=(\S+)', args).group(1)
    r = pd.read_parquet(f'{comp}/results/episode_{ep:06d}.parquet'); r = r[r.environment_id == r.environment_id.min()]
    src = FR7 % ep if ep in (15, 16) else FR % ep
    z = np.load(src); names = [str(n) for n in z['names']]
    for obj, g in r.groupby('object_body_name'):
        g = g.drop_duplicates('reference_frame').sort_values('reference_frame'); Ref = T(g); fr = g.reference_frame.values
        W = z['world_from_object'][fr, names.index(obj)]
        A = Ref[0] @ np.linalg.inv(W[0]); P = A @ W
        dp = np.linalg.norm(P[:, :3, 3] - Ref[:, :3, 3], axis=1) * 100
        dr = np.degrees(np.linalg.norm(R.from_matrix(np.einsum('nji,njk->nik', P[:, :3, :3], Ref[:, :3, :3])).as_rotvec(), axis=1))
        print(f'ep{ep:2d} {obj:14s} n={len(fr)} frames {fr[0]}-{fr[-1]} (src T={len(z["world_from_object"])}) max dpos {dp.max():.3f} cm  max drot {dr.max():.3f} deg  | {comp.split("campaign_v16_20261023/")[-1][:60]}')
