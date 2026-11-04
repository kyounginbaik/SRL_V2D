"""v14 = v13 + episode 19 sweeper face relabel: achieved sweeper poses T -> T @ G (G = half turn about the sweeper's
long axis, a symmetry of the submitted mesh), mesh bytes unchanged so the official registration is identical. Physical
placement of the symmetric body is unchanged; only which face is reported changes (chosen from the video marker-ball cue,
see recon_v15_20261022/episode_000019/change.json). Re-packs with the unchanged official packer and re-validates."""
import json, shutil, subprocess, sys, hashlib, importlib.util
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R
SRC = Path('/data1/SRL_V2D/challenge_submissions/track3_v13_20261022/all_20_episodes'); DST = Path('/data1/SRL_V2D/challenge_submissions/track3_v14_20261022/all_20_episodes')
CH = Path('/data1/SRL_V2D/challenge_reconstruction/recon_v15_20261022/episode_000019/change.json')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
if DST.exists(): raise SystemExit('exists')
shutil.copytree(SRC, DST, ignore=shutil.ignore_patterns('__pycache__')); kit = DST / 'submission_kit'
ch = json.loads(CH.read_text()); G = np.asarray(ch['G'], float)
p = DST / 'results/episode_000019.parquet'; t = pq.read_table(p); meta = t.schema.metadata; df = t.to_pandas(); before = df.copy()
rows = df.object_body_name == 'mini_sweeper'
P = df.loc[rows, ['achieved_position_x', 'achieved_position_y', 'achieved_position_z']].to_numpy(float)
Q = df.loc[rows, ['achieved_quaternion_x', 'achieved_quaternion_y', 'achieved_quaternion_z', 'achieved_quaternion_w']].to_numpy(float)
Rm = R.from_quat(Q).as_matrix(); Rn = Rm @ G[:3, :3]; Pn = P + np.einsum('nij,j->ni', Rm, G[:3, 3])
df.loc[rows, ['achieved_position_x', 'achieved_position_y', 'achieved_position_z']] = Pn
df.loc[rows, ['achieved_quaternion_x', 'achieved_quaternion_y', 'achieved_quaternion_z', 'achieved_quaternion_w']] = R.from_matrix(Rn).as_quat()
pq.write_table(pa.Table.from_pandas(df, preserve_index=False).replace_schema_metadata(meta), p)
print('sweeper rows changed:', int(rows.sum()), 'mean origin shift cm', float(np.linalg.norm(Pn - P, axis=1).mean() * 100))
build = json.load(open('/data1/SRL_V2D/challenge_reconstruction/episode37_ilchold_submission_20261019/build_manifest.json'))
(DST / 'track3_submission.parquet').unlink(); (DST / 'track3_submission.cd-o.parquet').unlink(missing_ok=True)
log = DST / 'preparation'; log.mkdir(exist_ok=True)
for name, cmd in (('pack_v14.log', [sys.executable, '-B', 'tools/pack_submission.py', '--track', 'track_3', '--results', str(DST / 'results'), '--meshes', str(DST / 'meshes'), '--code-commit', build['code_commit_url'], '-o', str(DST / 'track3_submission.parquet')]),
                  ('preflight_offline_v14.log', [sys.executable, '-B', 'tools/preflight.py', '--track', 'track_3', '--results', str(DST / 'results'), '--skip-kaggle'])):
    r = subprocess.run(cmd, cwd=kit, capture_output=True, text=True); (log / name).write_text(r.stdout + r.stderr)
    if r.returncode: raise SystemExit(name + ' failed\n' + r.stdout[-1500:] + r.stderr[-1500:])
spec = importlib.util.spec_from_file_location('up', kit / 'submit/v2d_submit.py'); up = importlib.util.module_from_spec(spec); spec.loader.exec_module(up)
f = up.validate_submission(DST / 'track3_submission.parquet', 'track_3', json.load(open(kit / 'config/tracks.json')))
o = pd.read_parquet(SRC / 'track3_submission.parquet'); eo = o.row_id.str[9:15].astype(int); en = f.row_id.str[9:15].astype(int)
cols = ['row_id', 'pos_x', 'pos_y', 'pos_z', 'quat_x', 'quat_y', 'quat_z', 'quat_w', 'checkpoint_sha256', 'object_mesh_b64']
diff = [e for e in sorted(eo.unique()) if not o.loc[eo == e, cols].sort_values('row_id').reset_index(drop=True).equals(f.loc[en == e, cols].sort_values('row_id').reset_index(drop=True))]
mesh_same = o.loc[eo == 19, ['row_id', 'object_mesh_b64']].sort_values('row_id').reset_index(drop=True).equals(f.loc[en == 19, ['row_id', 'object_mesh_b64']].sort_values('row_id').reset_index(drop=True))
print('rows', len(f), 'episodes differing from v13:', diff, '| ep19 meshes byte-identical:', mesh_same)
if diff != [19] or not mesh_same: raise SystemExit('unexpected change set')
m = json.load(open(DST / 'manifest.json')); m['created_utc'] = datetime.now(timezone.utc).isoformat()
m['episode_19_sweeper_face_relabel'] = dict(change=str(CH), G=ch['G'], evidence=ch['evidence'], note='achieved mini_sweeper poses right-multiplied by a half turn about its long axis (a symmetry of the unchanged submitted mesh); robot actions, checkpoint and physics unchanged; face chosen from the video marker-ball cue (public GT validation in public_gt_check_20261020/sweeper_results.md)')
m['combined_submission'] = dict(path='track3_submission.parquet', sha256=sha(DST / 'track3_submission.parquet')); json.dump(m, open(DST / 'manifest.json', 'w'), indent=2)
(DST / 'README.md').write_text((DST / 'README.md').read_text() + '''
## Episode 19 sweeper face (v14)

The sweeper looks the same after a half turn about its long axis except for the marker side. Its reported poses were
relabelled by that half turn (a symmetry of the unchanged submitted mesh) to the face chosen from the video. The robot
rollout, actions and checkpoint are unchanged. Not uploaded.
''')
for d in DST.rglob('__pycache__'): shutil.rmtree(d)
inv = {str(q.relative_to(DST)): sha(q) for q in sorted(DST.rglob('*')) if q.is_file() and q.name != 'SHA256SUMS.json'}
json.dump(inv, open(DST / 'SHA256SUMS.json', 'w'), indent=2)
print(json.dumps(dict(output=str(DST), parquet_sha256=m['combined_submission']['sha256'])))
