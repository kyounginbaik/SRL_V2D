"""Stream P ep33 (phase 6): build a 1-episode Track 3 submission parquet for score_vs_compose.py, same mapping as Stream R's
make_sub32.py / the v21 packer: row_id track_3/e000033/w000/fFFFFFF/oO, pose = achieved_* of the results parquet, object_mesh_b64 +
provenance columns copied from the v21 package rows of ep33 (meshes 57bb269a cup / c9ab88e3 spoon; checked against the component).
Mode 'tag':   make_sub33.py tag TAG_EPISODE_DIR OUT_DIR
Mode 'poses': make_sub33.py poses POSES_SOURCEWORLD.npz OUT_DIR   (diagnostic: a fake rollout whose achieved poses are given
              world_from_object (N,2,4,4) in the final_recon world, our body frames; used to validate a converted reference)"""
import sys, shutil, hashlib, numpy as np, pandas as pd
from pathlib import Path
from scipy.spatial.transform import Rotation as R
mode, src, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True); (out / 'results').mkdir(exist_ok=True)
V21 = '/data1/SRL_V2D/challenge_submissions/track3_v21_20261024/all_20_episodes_post/track3_submission.parquet'
TEMPLATE = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates/P_ep33_regrasp_cem3/episode_000033')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
s = pd.read_parquet(V21); e = s[s.row_id.str.contains('/e000033/')].set_index('row_id')
if mode == 'tag':
    for name, pre in (('blue_cup', '57bb269a06a4'), ('wooden_spoon', 'c9ab88e33e5c')):
        m = next((src / 'meshes').rglob(f'{name}.obj')) if (src / 'meshes').exists() else None
        if m is not None: assert sha(m).startswith(pre), (name, sha(m))
    r = pd.read_parquet(src / 'results/episode_000033.parquet')
else:
    r = pd.read_parquet(TEMPLATE / 'results/episode_000033.parquet').copy(); W = np.load(src)['world_from_object']
    for i in range(len(r)):
        k, o = int(r.step.iloc[i]), int(r.object_id.iloc[i]); T = W[min(k, len(W) - 1), o]; q = R.from_matrix(T[:3, :3]).as_quat()
        for j, c in enumerate('xyz'): r.loc[r.index[i], f'achieved_position_{c}'] = T[j, 3]; r.loc[r.index[i], f'achieved_quaternion_{c}'] = q[j]
        r.loc[r.index[i], 'achieved_quaternion_w'] = q[3]
rows = []
for _, x in r.sort_values(['step', 'object_id']).iterrows():
    rid = f"track_3/e000033/w000/f{int(x.step):06d}/o{int(x.object_id)}"; t = e.loc[rid]
    rows.append(dict(row_id=rid, pos_x=float(x.achieved_position_x), pos_y=float(x.achieved_position_y), pos_z=float(x.achieved_position_z),
                     quat_x=float(x.achieved_quaternion_x), quat_y=float(x.achieved_quaternion_y), quat_z=float(x.achieved_quaternion_z), quat_w=float(x.achieved_quaternion_w),
                     **{c: t[c] for c in e.columns if c not in ('pos_x', 'pos_y', 'pos_z', 'quat_x', 'quat_y', 'quat_z', 'quat_w')}))
pd.DataFrame(rows).to_parquet(out / 'submission.parquet', index=False); r.to_parquet(out / 'results/episode_000033.parquet', index=False)
print('rows', len(rows))
