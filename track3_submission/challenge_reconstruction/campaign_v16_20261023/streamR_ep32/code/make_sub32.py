"""Stream R ep32: build a 1-episode Track 3 submission parquet from a verified component's results parquet
(common_v2_candidates/<TAG>/episode_000032/results/episode_000032.parquet): row_id track_3/e000032/w000/fFFFFFF/oO, pose = achieved_*
(same mapping as the v21 packer: verified on R_ep32_regrasp_power06 rows), object_mesh_b64 on the frame-0 row of each object copied from the
v21 package (ep32 meshes 3c3c5328 cup / c9ab88e3 spoon; the script checks the component ships those exact meshes); provenance columns copied.
Writes OUT_DIR/submission.parquet and OUT_DIR/results/episode_000032.parquet (for score_vs_compose.py).
usage: make_sub32.py TAG_DIR OUT_DIR"""
import sys, shutil, hashlib, numpy as np, pandas as pd
from pathlib import Path
tag, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True); (out / 'results').mkdir(exist_ok=True)
V21 = '/data1/SRL_V2D/challenge_submissions/track3_v21_20261024/all_20_episodes_post/track3_submission.parquet'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
for name, pre in (('blue_cup', '3c3c5328180b'), ('wooden_spoon', 'c9ab88e33e5c')):
    m = next((tag / 'meshes').rglob(f'{name}.obj')); assert sha(m).startswith(pre), (name, sha(m))
s = pd.read_parquet(V21); e = s[s.row_id.str.contains('/e000032/')].set_index('row_id')
r = pd.read_parquet(tag / 'results/episode_000032.parquet'); rows = []
for _, x in r.sort_values(['step', 'object_id']).iterrows():
    rid = f"track_3/e000032/w000/f{int(x.step):06d}/o{int(x.object_id)}"; t = e.loc[rid]
    rows.append(dict(row_id=rid, pos_x=float(x.achieved_position_x), pos_y=float(x.achieved_position_y), pos_z=float(x.achieved_position_z),
                     quat_x=float(x.achieved_quaternion_x), quat_y=float(x.achieved_quaternion_y), quat_z=float(x.achieved_quaternion_z), quat_w=float(x.achieved_quaternion_w),
                     **{c: t[c] for c in e.columns if c not in ('pos_x', 'pos_y', 'pos_z', 'quat_x', 'quat_y', 'quat_z', 'quat_w')}))
pd.DataFrame(rows).to_parquet(out / 'submission.parquet', index=False); shutil.copy2(tag / 'results/episode_000032.parquet', out / 'results/episode_000032.parquet')
print('rows', len(rows))
