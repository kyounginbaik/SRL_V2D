"""Build a 1-episode submission parquet for ep36 from a common_v2_candidates run: v21 ep36 rows as template (provenance columns,
mesh b64 of our mesh c9ab88e33e5c on row 0), pos/quat = the run's achieved poses. usage: mk_sub.py RUN_EP_DIR OUT_DIR"""
import sys, os, shutil, pandas as pd, numpy as np
run, out = sys.argv[1:3]; os.makedirs(out + '/results', exist_ok=True)
s = pd.read_parquet('/data1/SRL_V2D/challenge_submissions/track3_v21_20261024/all_20_episodes_post/track3_submission.parquet')
x = s[s.row_id.str.contains('/e000036/')].copy().sort_values('row_id').reset_index(drop=True)
r = pd.read_parquet(run + '/results/episode_000036.parquet'); r = r[r.environment_id == 0].sort_values('reference_frame').reset_index(drop=True)
assert len(r) == len(x) == 407
for a in 'xyz': x['pos_' + a] = r['achieved_position_' + a].values
for a in 'xyzw': x['quat_' + a] = r['achieved_quaternion_' + a].values
x.to_parquet(out + '/sub.parquet'); shutil.copy2(run + '/results/episode_000036.parquet', out + '/results/episode_000036.parquet'); print('ok', out)
