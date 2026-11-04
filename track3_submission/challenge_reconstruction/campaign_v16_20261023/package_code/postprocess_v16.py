#!/usr/bin/env python3
"""Package-level postprocessing carried over from v14/v15 onto a freshly assembled twenty-episode package:
  (1) episode 19: achieved mini_sweeper poses right-multiplied by the constant body relabel G (half turn about the sweeper's
      long axis, a symmetry of the unchanged submitted mesh; face chosen from the video marker-ball cue, recon_v15_20261022/episode_000019/change.json);
  (2) episode 15: submitted plastic_dish_rack mesh replaced by the scan-derived rack in the same body frame (mesh-only, as in v8+).
Writes a NEW package directory, re-packs with the unchanged official packer, re-validates, and (optionally) checks the changed
rows/meshes against a reference package. Robot actions, checkpoints and physics are unchanged. Nothing is uploaded.
usage: postprocess_v16.py SRC_PACKAGE DST_PACKAGE [--verify-against PACKAGE] [--skip-ep15-mesh]"""
import argparse, hashlib, importlib.util, json, shutil, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R
CHANGE = Path('/data1/SRL_V2D/challenge_reconstruction/recon_v15_20261022/episode_000019/change.json')
RACK = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/ep15_v8/collision_recon/episode_000015/assets/plastic_dish_rack_visual.obj')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
ap = argparse.ArgumentParser(); ap.add_argument('src', type=Path); ap.add_argument('dst', type=Path); ap.add_argument('--verify-against', type=Path); ap.add_argument('--skip-ep15-mesh', action='store_true')
a = ap.parse_args(); SRC, DST = a.src.resolve(), a.dst.resolve()
if DST.exists(): raise SystemExit('exists ' + str(DST))
shutil.copytree(SRC, DST, ignore=shutil.ignore_patterns('__pycache__', '*.pyc')); kit = DST / 'submission_kit'
# (1) episode 19 relabel
change = json.loads(CHANGE.read_text()); G = np.asarray(change['G']); obj = change['object']
p19 = DST / 'results/episode_000019.parquet'; t = pq.read_table(p19); df = t.to_pandas(); m = df.object_body_name == obj
q = df.loc[m, ['achieved_quaternion_x', 'achieved_quaternion_y', 'achieved_quaternion_z', 'achieved_quaternion_w']].to_numpy(); p = df.loc[m, ['achieved_position_x', 'achieved_position_y', 'achieved_position_z']].to_numpy()
T = np.tile(np.eye(4), (len(q), 1, 1)); T[:, :3, :3] = R.from_quat(q).as_matrix(); T[:, :3, 3] = p; T2 = T @ G
df.loc[m, ['achieved_position_x', 'achieved_position_y', 'achieved_position_z']] = T2[:, :3, 3]; df.loc[m, ['achieved_quaternion_x', 'achieved_quaternion_y', 'achieved_quaternion_z', 'achieved_quaternion_w']] = R.from_matrix(T2[:, :3, :3]).as_quat()
old19 = sha(p19); pq.write_table(pa.Table.from_pandas(df, preserve_index=False).replace_schema_metadata(t.schema.metadata), p19)
# (2) episode 15 mesh
rec15 = None
if not a.skip_ep15_mesh:
    tgt = DST / 'meshes/episode_000015/plastic_dish_rack.obj'; rec15 = dict(mesh='meshes/episode_000015/plastic_dish_rack.obj', previous_sha256=sha(tgt), source=str(RACK), source_sha256=sha(RACK)); shutil.copy2(RACK, tgt); rec15['new_sha256'] = sha(tgt)
# repack
build = json.loads((DST / 'evidence/episode_000001/build_manifest.json').read_text())
(DST / 'track3_submission.parquet').unlink(); (DST / 'track3_submission.cd-o.parquet').unlink(missing_ok=True); log = DST / 'preparation'; log.mkdir(exist_ok=True)
for name, cmd in (('pack_post.log', [sys.executable, 'tools/pack_submission.py', '--track', 'track_3', '--results', str(DST / 'results'), '--meshes', str(DST / 'meshes'), '--code-commit', build['code_commit_url'], '-o', str(DST / 'track3_submission.parquet')]),
                  ('preflight_offline_post.log', [sys.executable, 'tools/preflight.py', '--track', 'track_3', '--results', str(DST / 'results'), '--skip-kaggle'])):
    r = subprocess.run(cmd, cwd=kit, capture_output=True, text=True); (log / name).write_text(r.stdout + r.stderr)
    if r.returncode: raise SystemExit(name + ' failed:\n' + r.stdout[-2000:] + r.stderr[-2000:])
spec = importlib.util.spec_from_file_location('up', kit / 'submit/v2d_submit.py'); up = importlib.util.module_from_spec(spec); spec.loader.exec_module(up)
frame = up.validate_submission(DST / 'track3_submission.parquet', 'track_3', json.loads((kit / 'config/tracks.json').read_text()))
before = pd.read_parquet(SRC / 'track3_submission.parquet'); cols = ['row_id', 'pos_x', 'pos_y', 'pos_z', 'quat_x', 'quat_y', 'quat_z', 'quat_w']
ep = frame.row_id.str.extract(r'/e(\d{6})/')[0].astype(int); slot = frame.row_id.str.extract(r'/o(\d)$')[0].astype(int)
bep = before.row_id.str.extract(r'/e(\d{6})/')[0].astype(int); bslot = before.row_id.str.extract(r'/o(\d)$')[0].astype(int)
keep_b = ~((bep == 19) & (bslot == change['slot'])); keep_a = ~((ep == 19) & (slot == change['slot']))
if len(frame) != 9120 or not before.loc[keep_b, cols].sort_values('row_id').reset_index(drop=True).equals(frame.loc[keep_a, cols].sort_values('row_id').reset_index(drop=True)): raise SystemExit('rows other than the ep19 sweeper changed')
for c in ('checkpoint_sha256', 'eval_script_sha256', 'eval_code_sha256', 'code_commit_url'):
    if not before[c].sort_values().reset_index(drop=True).equals(frame[c].sort_values().reset_index(drop=True)): raise SystemExit('provenance column changed: ' + c)
report = dict(ep19_relabel_rows=int(m.sum()))
if a.verify_against:
    ref = pd.read_parquet(a.verify_against / 'track3_submission.parquet'); rep = ref.row_id.str.extract(r'/e(\d{6})/')[0].astype(int); rslot = ref.row_id.str.extract(r'/o(\d)$')[0].astype(int)
    x = ref.loc[(rep == 19) & (rslot == change['slot']), cols].sort_values('row_id').reset_index(drop=True); y = frame.loc[(ep == 19) & (slot == change['slot']), cols].sort_values('row_id').reset_index(drop=True)
    d = np.abs(x[cols[1:]].to_numpy() - y[cols[1:]].to_numpy()).max(); report['ep19_sweeper_max_abs_diff_vs_reference'] = float(d)
    if d > 1e-9: raise SystemExit(f'ep19 relabelled rows differ from the reference package by {d}')
    if rec15:
        rm = ref.loc[(rep == 15), ['row_id', 'object_mesh_b64']].sort_values('row_id').reset_index(drop=True); fm = frame.loc[(ep == 15), ['row_id', 'object_mesh_b64']].sort_values('row_id').reset_index(drop=True)
        if not rm.equals(fm): raise SystemExit('ep15 packed meshes differ from the reference package')
        report['ep15_mesh_matches_reference'] = True
man = json.loads((DST / 'manifest.json').read_text()); man['created_utc'] = datetime.now(timezone.utc).isoformat()
man['episode_19_sweeper_face_relabel'] = dict(change=str(CHANGE), change_sha256=sha(CHANGE), G=change['G'], evidence=change['evidence'], previous_results_sha256=old19, new_results_sha256=sha(p19),
    note='achieved mini_sweeper poses right-multiplied by a half turn about its long axis (a symmetry of the unchanged submitted mesh); robot actions, checkpoint and physics unchanged; face chosen from the video marker-ball cue (public GT validation in public_gt_check_20261020/sweeper_results.md)')
if rec15: man['episode_15_mesh_only'] = dict(rec15, note='physics trajectory, robot actions and checkpoint unchanged; only the submitted rack mesh is the scan rack in the same rack body frame (carried over from v8-v15).')
man['combined_submission'] = dict(path='track3_submission.parquet', sha256=sha(DST / 'track3_submission.parquet')); man['postprocess'] = dict(script_sha256=sha(__file__), source_package=str(SRC), report=report)
(DST / 'manifest.json').write_text(json.dumps(man, indent=2) + '\n'); shutil.copy2(__file__, log / Path(__file__).name)
(DST / 'README.md').write_text((DST / 'README.md').read_text() + '''
## Episode 19 sweeper face

The sweeper looks the same after a half turn about its long axis except for the marker side. Its reported poses were
relabelled by that half turn (a symmetry of the unchanged submitted mesh) to the face chosen from the video. The robot
rollout, actions and checkpoint are unchanged.
''' + ('''
## Episode 15 rack mesh (mesh-only)

Episode 15 keeps its physics trajectory, robot actions and checkpoint. Only its submitted plastic_dish_rack mesh is the
scan-derived rack expressed in the same rack body frame, so the poses remain valid.
''' if rec15 else '') + '\nThe combined Parquet was re-packed with the unchanged official packer; offline preflight and the uploader local validation passed. Not uploaded.\n')
inv = {str(p.relative_to(DST)): sha(p) for p in sorted(DST.rglob('*')) if p.is_file() and p != DST / 'SHA256SUMS.json'}
(DST / 'SHA256SUMS.json').write_text(json.dumps(inv, indent=2) + '\n')
print(json.dumps(dict(output=str(DST), parquet_sha256=man['combined_submission']['sha256'], rows=len(frame), report=report, uploaded=False)))
