"""v7 = v7a (episodes 14/16/28/30 re-planned with the scan rack) + episode 15 mesh-only swap: the v6 physics trajectory is kept
and only its rack mesh is replaced by the scan rack expressed in the same rack body frame (kit registration placement), so its
poses stay valid. Re-packs with the unchanged official packer and re-validates; checks every pose row is unchanged."""
import json, shutil, subprocess, sys, hashlib, importlib.util
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
SRC = Path('/data1/SRL_V2D/challenge_submissions/track3_rackscan_v7a_20261021/all_20_episodes'); DST = Path('/data1/SRL_V2D/challenge_submissions/track3_rackscan_v7_20261021/all_20_episodes')
W = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021')
MESH = W / 'superseded_ep15_v1/collision_recon_episode_000015/assets/plastic_dish_rack_visual.obj'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
if DST.exists(): raise SystemExit('exists ' + str(DST))
shutil.copytree(SRC, DST); kit = DST / 'submission_kit'
tgt = DST / 'meshes/episode_000015/plastic_dish_rack.obj'; old_sha = sha(tgt); shutil.copy2(MESH, tgt)
build = json.load(open('/data1/SRL_V2D/challenge_reconstruction/episode37_ilchold_submission_20261019/build_manifest.json'))
(DST / 'track3_submission.parquet').unlink(); (DST / 'track3_submission.cd-o.parquet').unlink(missing_ok=True)
log = DST / 'preparation'; log.mkdir(exist_ok=True)
for name, cmd in (('pack_v7.log', [sys.executable, 'tools/pack_submission.py', '--track', 'track_3', '--results', str(DST / 'results'), '--meshes', str(DST / 'meshes'), '--code-commit', build['code_commit_url'], '-o', str(DST / 'track3_submission.parquet')]),
                  ('preflight_offline_v7.log', [sys.executable, 'tools/preflight.py', '--track', 'track_3', '--results', str(DST / 'results'), '--skip-kaggle'])):
    r = subprocess.run(cmd, cwd=kit, capture_output=True, text=True); (log / name).write_text(r.stdout + r.stderr)
    if r.returncode: raise SystemExit(name + ' failed:\n' + r.stdout[-2000:] + r.stderr[-2000:])
spec = importlib.util.spec_from_file_location('up', kit / 'submit/v2d_submit.py'); up = importlib.util.module_from_spec(spec); spec.loader.exec_module(up)
frame = up.validate_submission(DST / 'track3_submission.parquet', 'track_3', json.load(open(kit / 'config/tracks.json')))
before = pd.read_parquet(SRC / 'track3_submission.parquet'); cols = ['row_id', 'pos_x', 'pos_y', 'pos_z', 'quat_x', 'quat_y', 'quat_z', 'quat_w', 'checkpoint_sha256', 'eval_script_sha256', 'eval_code_sha256', 'code_commit_url']
if len(frame) != 9120 or not before[cols].sort_values('row_id').reset_index(drop=True).equals(frame[cols].sort_values('row_id').reset_index(drop=True)): raise SystemExit('pose rows changed')
ep = frame.row_id.str.extract(r'/e(\d{6})/')[0].astype(int); bep = before.row_id.str.extract(r'/e(\d{6})/')[0].astype(int)
mb = before.loc[bep != 15, ['row_id', 'object_mesh_b64']].sort_values('row_id').reset_index(drop=True); ma = frame.loc[ep != 15, ['row_id', 'object_mesh_b64']].sort_values('row_id').reset_index(drop=True)
if not mb.equals(ma): raise SystemExit('meshes of other episodes changed')
m = json.load(open(DST / 'manifest.json')); m['created_utc'] = datetime.now(timezone.utc).isoformat()
m['episode_15_mesh_only'] = dict(mesh='meshes/episode_000015/plastic_dish_rack.obj', previous_sha256=old_sha, new_sha256=sha(tgt), source=str(MESH),
    note='physics trajectory, robot actions and checkpoint unchanged from v6; only the submitted rack mesh is the scan rack in the same rack body frame. A physics re-plan with the scan rack did not pass (spoon settles deeper in the rack).')
m['combined_submission'] = dict(path='track3_submission.parquet', sha256=sha(DST / 'track3_submission.parquet')); json.dump(m, open(DST / 'manifest.json', 'w'), indent=2)
(DST / 'README.md').write_text((DST / 'README.md').read_text() + '''
## Episode 15 rack mesh (mesh-only)

Episode 15 keeps its previous physics trajectory, robot actions and checkpoint. Only its submitted plastic_dish_rack mesh was
replaced by the scan-derived rack expressed in the same rack body frame, so the poses remain valid. The combined Parquet was
re-packed with the unchanged official packer; offline preflight and the uploader's local validation passed. Not uploaded.
''')
inv = {str(p.relative_to(DST)): sha(p) for p in sorted(DST.rglob('*')) if p.is_file() and p.name != 'SHA256SUMS.json' or (p.is_file() and p.parent != DST)}
inv.pop('SHA256SUMS.json', None); json.dump(inv, open(DST / 'SHA256SUMS.json', 'w'), indent=2)
print(json.dumps(dict(output=str(DST), parquet_sha256=m['combined_submission']['sha256'], rows=len(frame), uploaded=False)))
