"""Build a 1-episode (ep19) Track-3 submission parquet from a verified component's results parquet (row format + packed meshes
taken from the v21 pre-postprocess package rows of episode 19; meshes identical: cf7bf340 / 1d298c01), optionally apply the
package sweeper relabel G (postprocess_v16), then score with compose_target/score_vs_compose.py.
usage: score_component.py COMPONENT_OUTPUT_DIR(common_v2_candidates/<TAG>/episode_000019) OUT_DIR [--relabel]"""
import sys, json, subprocess, hashlib, shutil, numpy as np, pandas as pd
from pathlib import Path
from scipy.spatial.transform import Rotation as R
comp, out = Path(sys.argv[1]), Path(sys.argv[2]); relabel = '--relabel' in sys.argv
C16 = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023')
PKG = Path('/data1/SRL_V2D/challenge_submissions/track3_v21_20261024/all_20_episodes/track3_submission.parquet')
out.mkdir(parents=True, exist_ok=True); (out / 'results').mkdir(exist_ok=True)
res = pd.read_parquet(comp / 'results/episode_000019.parquet'); shutil.copy2(comp / 'results/episode_000019.parquet', out / 'results/episode_000019.parquet')
s = pd.read_parquet(PKG); s = s[s.row_id.str.contains('/e000019/')].copy().reset_index(drop=True)
f = s.row_id.str.extract(r'/f(\d{6})/')[0].astype(int); o = s.row_id.str.extract(r'/o(\d)$')[0].astype(int)
idx = res.set_index(['reference_frame', 'object_id'])
P = idx.loc[list(zip(f, o)), ['achieved_position_x', 'achieved_position_y', 'achieved_position_z']].to_numpy()
Q = idx.loc[list(zip(f, o)), ['achieved_quaternion_x', 'achieved_quaternion_y', 'achieved_quaternion_z', 'achieved_quaternion_w']].to_numpy()
if relabel:
    ch = json.loads(Path('/data1/SRL_V2D/challenge_reconstruction/recon_v15_20261022/episode_000019/change.json').read_text()); G = np.array(ch['G']); m = (o == ch['slot']).to_numpy()
    T = np.tile(np.eye(4), (m.sum(), 1, 1)); T[:, :3, :3] = R.from_quat(Q[m]).as_matrix(); T[:, :3, 3] = P[m]; T = T @ G; P[m] = T[:, :3, 3]; Q[m] = R.from_matrix(T[:, :3, :3]).as_quat()
s[['pos_x', 'pos_y', 'pos_z']] = P; s[['quat_x', 'quat_y', 'quat_z', 'quat_w']] = Q
cj = json.loads((comp / 'completed.json').read_text()); s['checkpoint_sha256'] = cj['checkpoint']['sha256']
s.to_parquet(out / 'submission_ep19.parquet', index=False)
r = subprocess.run([sys.executable, str(C16 / 'compose_target/score_vs_compose.py'), str(out / 'submission_ep19.parquet'), str(out / 'results'), str(C16 / 'compose_target'), str(out / 'score_vs_compose.json'), '19'], capture_output=True, text=True)
print(r.stdout[-1500:], r.stderr[-1500:])
json.dump(dict(component=str(comp), relabel_applied=relabel, rows=len(s), checkpoint_sha256=cj['checkpoint']['sha256'], row_template=str(PKG), score=json.loads((out / 'score_vs_compose.json').read_text())), open(out / 'record.json', 'w'), indent=1)
