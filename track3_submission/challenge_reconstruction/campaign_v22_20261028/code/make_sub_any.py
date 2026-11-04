"""Build a 1-episode Track 3 submission parquet from any build-v2 component output dir
(<TAG_DIR>/results/episode_XXXXXX.parquet + <TAG_DIR>/meshes/**/<object>.obj). Pose = achieved_* of the first environment
(same mapping as the v21 packer, from campaign_v16 streamT make_sub.py); provenance columns copied from the v21 package;
object_mesh_b64 encoded from the component's OWN mesh with the kit helper (budget 4096/4096).
usage: make_sub_any.py EP TAG_DIR OUT_DIR"""
import sys, shutil, pandas as pd, numpy as np
from pathlib import Path
sys.path.insert(0, '/data1/SRL_V2D/challenge_submissions_SRL/analysis_20261020/scripts')
import public_kit_score as K
ep, tag, out = int(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]); (out / 'results').mkdir(parents=True, exist_ok=True)
V21 = '/data1/SRL_V2D/challenge_submissions/track3_v21_20261024/all_20_episodes_post/track3_submission.parquet'
s = pd.read_parquet(V21); e = s[s.row_id.str.contains(f'/e{ep:06d}/')].set_index('row_id')
r = pd.read_parquet(tag / f'results/episode_{ep:06d}.parquet'); r = r[r.environment_id == r.environment_id.min()]
names = r.drop_duplicates('object_id').sort_values('object_id')[['object_id', 'object_body_name']].values.tolist()
payload = {}
for oid, name in names:
    m = next((tag / 'meshes').rglob(f'{name}.obj')); v, f = K.budget_mesh(str(m), 4096, 4096)
    payload[int(oid)] = K.encode(vertices=np.asarray(v, np.float64), faces=np.asarray(f, np.float64))
prov = [c for c in e.columns if c not in ('pos_x', 'pos_y', 'pos_z', 'quat_x', 'quat_y', 'quat_z', 'quat_w', 'object_mesh_b64')]
rows = []
for _, x in r.sort_values(['step', 'object_id']).iterrows():
    rid = f"track_3/e{ep:06d}/w000/f{int(x.step):06d}/o{int(x.object_id)}"; t = e.loc[rid]
    rows.append(dict(row_id=rid, pos_x=float(x.achieved_position_x), pos_y=float(x.achieved_position_y), pos_z=float(x.achieved_position_z),
                     quat_x=float(x.achieved_quaternion_x), quat_y=float(x.achieved_quaternion_y), quat_z=float(x.achieved_quaternion_z),
                     quat_w=float(x.achieved_quaternion_w), **{c: t[c] for c in prov},
                     object_mesh_b64=payload[int(x.object_id)] if int(x.step) == 0 else ''))
assert len(rows) == len(e), (len(rows), len(e))
pd.DataFrame(rows).to_parquet(out / 'submission.parquet', index=False)
shutil.copy2(tag / f'results/episode_{ep:06d}.parquet', out / f'results/episode_{ep:06d}.parquet')
