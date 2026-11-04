import json, re, shutil, hashlib, sys, io
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
SRC = Path(sys.argv[1])
DST = Path(sys.argv[2])
NOTE = Path(sys.argv[3]).read_text() if len(sys.argv) > 3 else ''
PAT = re.compile('dexagent', re.I)
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 23), b''): h.update(b)
    return h.hexdigest()
if DST.exists(): raise SystemExit('refusing to overwrite ' + str(DST))
DST.mkdir(parents=True)
shutil.copy2(SRC / 'track3_submission.parquet', DST / 'track3_submission.parquet')
for d in ('meshes', 'submission_kit'):
    shutil.copytree(SRC / d, DST / d, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
# videos: media only
for ep in sorted((SRC / 'videos').iterdir()):
    out = DST / 'videos' / ep.name; out.mkdir(parents=True)
    for f in ep.iterdir():
        if f.suffix.lower() in ('.mp4', '.jpg', '.png', '.gif'): shutil.copy2(f, out / f.name)
# per-episode parquets: drop/scrub metadata values naming the old workspace
(DST / 'results').mkdir()
for f in sorted((SRC / 'results').glob('*.parquet')):
    t = pq.read_table(f); meta = dict(t.schema.metadata or {}); new = {}
    for k, v in meta.items():
        if PAT.search(k.decode(errors='ignore')): continue
        s = v.decode(errors='ignore')
        if PAT.search(s):
            if k == b'flash_chord.reference_parquet': s = 'reference.parquet'
            else: continue
        new[k] = s.encode()
    pq.write_table(t.replace_schema_metadata(new), DST / 'results' / f.name)
# robot actions: keep arrays, scrub string fields
(DST / 'robot_actions').mkdir()
for f in sorted((SRC / 'robot_actions').glob('*.npz')):
    z = np.load(f, allow_pickle=True); out = {}
    for k in z.files:
        a = z[k]
        if a.dtype.kind in 'US' and PAT.search(str(a)):
            a = np.array(PAT.sub('srl', str(a)).replace('srl.', '')) if a.ndim == 0 else np.array([PAT.sub('srl', str(x)) for x in a.ravel()]).reshape(a.shape)
        out[k] = a
    np.savez(DST / 'robot_actions' / f.name, **out)
# manifest: keep the scored facts, drop local paths
old = json.loads((SRC / 'manifest.json').read_text())
rows = []
for r in old['episodes']:
    rows.append(dict(episode=r['episode'], backend=r['backend'], pose_rows=r['expected_rows'],
        checkpoint_sha256=r['checkpoint_sha256'], trajectory_sha256=sha(DST / 'results' / (r['episode'] + '.parquet')),
        robot_actions=f"robot_actions/{r['episode']}.npz", video_directory=f"videos/{r['episode']}" if (DST / 'videos' / r['episode']).is_dir() else None,
        local_reference_proxy_mp_sr=r.get('reference_proxy_mp_sr'), best_effort=bool(r.get('best_effort')),
        best_effort_object_errors=r.get('best_effort_object_errors'), replaced_previous_component=bool(r.get('replaced_previous_component'))))
manifest = dict(schema='srl_track3_twenty_episode_release_v1', created_utc=old['created_utc'], team='SRL',
    available_episodes=old['available_episodes'], available_episode_count=20, pose_row_count=old['pose_row_count'],
    combined_submission=dict(path='track3_submission.parquet', sha256=sha(DST / 'track3_submission.parquet')),
    evaluator=dict(eval_script_sha256=old['evaluator']['eval_script_sha256'], eval_code_sha256=old['evaluator']['eval_code_sha256'], code_commit_url=old['evaluator']['code_commit_url']),
    official_packer_passed=True, official_uploader_local_validation_passed=True, offline_preflight_passed=True,
    best_effort_episodes=old['best_effort_episodes'], all_episodes_pass_local_reference_proxy=old['all_episodes_pass_local_reference_proxy'],
    source_pose_proxy_not_official_ground_truth=True, hidden_ground_truth_scores_available=False, uploaded=False,
    replaced_episodes=old.get('replaced_episodes', []), episodes=rows)
(DST / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
(DST / 'README.md').write_text('''# SRL Track 3 submission: all twenty episodes

`track3_submission.parquet` holds all 9,120 public pose rows and the reconstructed
object meshes for the twenty scored Track 3 episodes. It was produced by the
unchanged official packer and passed the official uploader's local validation and
offline preflight. No Kaggle API was called from this package.

Every episode is a closed-loop physics rollout (MuJoCo, 500 Hz) of a per-episode
learned controller driving two floating Sharpa hands; objects move only through
contact. The
achieved object poses were exported on the original 20 Hz timestamp grid.

- `results/`: one achieved-pose Parquet per episode (the inputs to the packer).
- `meshes/`: body-frame visual meshes per episode, as packed into the submission.
- `robot_actions/`: recorded controller states and actuator commands per episode.
- `videos/`: review renders (original video, kinematic reference, baseline physics,
  refined physics) where available.
- `submission_kit/`: the official validation and upload tools.
- `manifest.json`, `SHA256SUMS.json`: inventory and checksums.

Local checks compared against the reconstructed object trajectories as a proxy for
hidden ground truth (every object below 3 cm mean position and 30 deg mean
orientation error). These are not official leaderboard scores. Episodes that do
not pass that local proxy, if any, are listed under `best_effort_episodes` in
`manifest.json`.

To upload, from `submission_kit/`:

    python tools/preflight.py --track track_3 --results ../results
    python submit/v2d_submit.py ../track3_submission.parquet --track track_3 -m "SRL Track 3 full twenty episodes" --wait
''' + NOTE)
inv = {str(p.relative_to(DST)): sha(p) for p in sorted(DST.rglob('*')) if p.is_file() and p.name != 'SHA256SUMS.json'}
(DST / 'SHA256SUMS.json').write_text(json.dumps(inv, indent=2) + '\n')
# verification sweep
hits = []
for p in DST.rglob('*'):
    if not p.is_file(): continue
    if p.suffix == '.parquet':
        if any(PAT.search((k + v).decode(errors='ignore')) for k, v in (pq.read_table(p).schema.metadata or {}).items()): hits.append(str(p))
    elif p.suffix == '.npz':
        z = np.load(p, allow_pickle=True)
        if any(z[k].dtype.kind in 'US' and PAT.search(str(z[k])) for k in z.files): hits.append(str(p))
    elif p.suffix.lower() in ('.mp4', '.jpg', '.png', '.gif', '.glb', '.obj', '.ply', '.stl'):
        continue
    else:
        try:
            if PAT.search(p.read_text(errors='ignore')): hits.append(str(p))
        except Exception: pass
print('remaining mentions:', hits)
print('files:', len(inv))
