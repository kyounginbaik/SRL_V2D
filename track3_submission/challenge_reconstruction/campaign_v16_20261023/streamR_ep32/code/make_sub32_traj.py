"""Stream R ep32: PRE-SCREEN helper (not a deliverable): 1-episode submission parquet straight from an executed trajectory.npz (setup frame):
pose = source_from_setup @ trajectory pose (identical to the evaluator's results-parquet achieved pose, checked on R_ep32_regrasp_power06 to 1e-7),
meshes/provenance from the v21 package ep32 rows. Writes OUT/submission.parquet and a minimal OUT/results/episode_000032.parquet (object ids/names).
usage: make_sub32_traj.py TRAJ_NPZ OUT_DIR"""
import sys, json, numpy as np, pandas as pd
from pathlib import Path
from scipy.spatial.transform import Rotation as R
tr = np.load(sys.argv[1]); out = Path(sys.argv[2]); (out / 'results').mkdir(parents=True, exist_ok=True)
Ssrc = np.asarray(json.load(open('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/expanded_batch/repaired_export_inputs/episode_000032/input_manifest.json'))['source_from_setup'])
V21 = '/data1/SRL_V2D/challenge_submissions/track3_v21_20261024/all_20_episodes_post/track3_submission.parquet'
s = pd.read_parquet(V21); e = s[s.row_id.str.contains('/e000032/')].set_index('row_id')
names = [str(x) for x in tr['object_names']]; order = ['blue_cup', 'wooden_spoon']; rows = []; res = []
for k in range(len(tr['object_positions'])):
    for oid, n in enumerate(order):
        i = names.index(n); T = np.eye(4); T[:3, :3] = R.from_quat(tr['object_quaternions_wxyz'][k, i][[1, 2, 3, 0]]).as_matrix(); T[:3, 3] = tr['object_positions'][k, i]; Ts = Ssrc @ T; q = R.from_matrix(Ts[:3, :3]).as_quat()
        rid = f'track_3/e000032/w000/f{k:06d}/o{oid}'; t = e.loc[rid]
        rows.append(dict(row_id=rid, pos_x=Ts[0, 3], pos_y=Ts[1, 3], pos_z=Ts[2, 3], quat_x=q[0], quat_y=q[1], quat_z=q[2], quat_w=q[3], **{c: t[c] for c in e.columns if not c.startswith(('pos_', 'quat_'))}))
        res.append(dict(step=k, object_id=oid, object_body_name=n))
pd.DataFrame(rows).to_parquet(out / 'submission.parquet', index=False); pd.DataFrame(res).to_parquet(out / 'results/episode_000032.parquet', index=False); print('rows', len(rows))
