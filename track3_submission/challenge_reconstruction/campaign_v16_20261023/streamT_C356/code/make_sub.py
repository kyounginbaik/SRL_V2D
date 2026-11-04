"""Stream T (phase 6): build a 1-episode Track 3 submission parquet from a verified component's results parquet
(<TAG_DIR>/results/episode_XXXXXX.parquet): row_id track_3/eXXXXXX/w000/fFFFFFF/oO, pose = achieved_* (same mapping as the v21
packer, generalised from streamR_ep32/code/make_sub32.py); object_mesh_b64 + provenance columns copied from the v21 package rows
(the script checks the component ships the same meshes as v21: pot 8cbd35431829, spoon c9ab88e33e5c).
Writes OUT_DIR/submission.parquet and OUT_DIR/results/episode_XXXXXX.parquet (for score_vs_compose.py).
usage: make_sub.py EP TAG_DIR OUT_DIR"""
import sys, shutil, hashlib, pandas as pd
from pathlib import Path
ep, tag, out = int(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True); (out / 'results').mkdir(exist_ok=True)
V21 = '/data1/SRL_V2D/challenge_submissions/track3_v21_20261024/all_20_episodes_post/track3_submission.parquet'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
for name, pre in (('white_pot', '8cbd35431829'), ('wooden_spoon', 'c9ab88e33e5c')):
    m = next((tag / 'meshes').rglob(f'{name}.obj')); assert sha(m).startswith(pre), (name, sha(m))
s = pd.read_parquet(V21); e = s[s.row_id.str.contains(f'/e{ep:06d}/')].set_index('row_id')
r = pd.read_parquet(tag / f'results/episode_{ep:06d}.parquet'); r = r[r.environment_id == r.environment_id.min()]; rows = []
for _, x in r.sort_values(['step', 'object_id']).iterrows():
    rid = f"track_3/e{ep:06d}/w000/f{int(x.step):06d}/o{int(x.object_id)}"; t = e.loc[rid]
    rows.append(dict(row_id=rid, pos_x=float(x.achieved_position_x), pos_y=float(x.achieved_position_y), pos_z=float(x.achieved_position_z),
                     quat_x=float(x.achieved_quaternion_x), quat_y=float(x.achieved_quaternion_y), quat_z=float(x.achieved_quaternion_z), quat_w=float(x.achieved_quaternion_w),
                     **{c: t[c] for c in e.columns if c not in ('pos_x', 'pos_y', 'pos_z', 'quat_x', 'quat_y', 'quat_z', 'quat_w')}))
assert len(rows) == len(e), (len(rows), len(e))
pd.DataFrame(rows).to_parquet(out / 'submission.parquet', index=False); shutil.copy2(tag / f'results/episode_{ep:06d}.parquet', out / f'results/episode_{ep:06d}.parquet')
print('rows', len(rows))
