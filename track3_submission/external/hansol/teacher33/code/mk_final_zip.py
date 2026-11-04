"""Zip of the combined final Track 3 package (all 20 episodes): submission + CD-O parquet, per-episode build-v2 results
(18 from FINAL_20261104, ep32 from T32_T32c_b1, ep33 from R6M2), manifests, README, SHA256SUMS. Checks that each results parquet
reproduces that episode's pose rows of the submission exactly."""
import json, shutil, hashlib, zipfile, numpy as np, pandas as pd
from pathlib import Path
h = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
PK = Path('/data1/hansol/teacher33/packages/track3_FINAL_ep32T32cb1_ep33R6M2_20261105')
FIN = Path('/data1/SRL_V2D/challenge_submissions_SRL/FINAL_20261104/all_20_episodes')
R32 = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates/T32_T32c_b1/episode_000032/results/episode_000032.parquet')
R33 = Path('/data1/hansol/teacher33/packages/track3_FINAL_ep33T33_R6M2Cf0_r6moved_20261104/all_20_episodes/results/episode_000033.parquet')
NAME = 'track3_FINAL_ep32T32cb1_ep33R6M2_20261105'; O = Path('/data1/hansol/share') / NAME
if O.exists(): shutil.rmtree(O)
(O / 'results').mkdir(parents=True)
for f in ('track3_submission.parquet', 'track3_submission.cd-o.parquet'): shutil.copy2(PK / 'all_20_episodes' / f, O / f)
shutil.copy2(PK / 'COMBINED_PACKAGE.json', O / 'COMBINED_PACKAGE.json'); shutil.copy2(FIN / 'V32_BUILD.json', O / 'FINAL_20261104_V32_BUILD.json')
src = {}
for p in sorted((FIN / 'results').glob('episode_*.parquet')):
    s = R32 if p.name == 'episode_000032.parquet' else R33 if p.name == 'episode_000033.parquet' else p
    shutil.copy2(s, O / 'results' / p.name); src[p.name] = str(s)
sub = pd.read_parquet(O / 'track3_submission.parquet'); bad = []
for p in sorted((O / 'results').glob('episode_*.parquet')):
    ep = int(p.stem.split('_')[1]); r = pd.read_parquet(p); r = r[r.environment_id == r.environment_id.min()].sort_values(['step', 'object_id'])
    rid = [f'track_3/e{ep:06d}/w000/f{int(a):06d}/o{int(b)}' for a, b in zip(r.step, r.object_id)]
    s = sub.set_index('row_id').loc[rid]
    d = max(np.abs(s[c].to_numpy(float) - r[f'achieved_position_{c[-1]}'].to_numpy(float)).max() for c in ('pos_x', 'pos_y', 'pos_z'))
    if d > 0: bad.append((p.name, d))
print('results vs submission pose rows: max |diff| > 0 for', bad if bad else 'none (all 20 exact)')
(O / 'RESULTS_SOURCES.json').write_text(json.dumps(src, indent=1))
info = json.loads((O / 'COMBINED_PACKAGE.json').read_text())
(O / 'README.md').write_text(f"""# Track 3 final candidate: FINAL_20261104 + ep32 T32c_b1 + ep33 R6M2 (2026-11-05, Hansol)

- `track3_submission.parquet` sha256 {info['sha']}
- `track3_submission.cd-o.parquet` sha256 {info['cdo_sha']}
- Base: FINAL_20261104 ({info['final_sha'][:16]}). Only the ep32 rows (518, from the team's T32c_b1 package {info['ep32_sha'][:16]}, MP-SR 0.95 alone)
  and ep33 rows (696, from R6M2 {info['ep33_sha'][:16]}, MP-SR 0.95 alone) differ. Changed columns: pos/quat + checkpoint_sha256. Mesh cells = FINAL's.
- Provenance columns (eval_script/eval_code/packer/uploader hashes, code_commit_url) are uniform over all 20 episodes.
- Kit offline validate_submission OK (9120 rows, 20 episodes); CD-O derivation + validation OK. Nothing uploaded from this zip.
- `results/episode_*.parquet`: build-v2 rollouts per episode (sources in RESULTS_SOURCES.json); each reproduces its episode's pose rows exactly.
- ep33 R6M2: r6 reconstruction (flipped spoon, human-in-the-loop bowl marks; see EP33_AGENT_SPOON_MARKS.md), scene start moved to the
  reconstruction's frame 0, new teacher (firmer spoon grip), both-hands window beam. vs r6: cup 0.65 cm / 3.4 deg, spoon 0.83 cm / 7.8 deg;
  perturbation 12/16 pass, spoon lost 0/16. Known cosmetic issue: the right hand releases the spoon ~0.5 s early (14.1 s).
- Not included: submission_kit (use the official kit), the new CD-O meshes (this file keeps FINAL's mesh cells).
""")
lines = [f'{h(p)}  {p.relative_to(O)}' for p in sorted(O.rglob('*')) if p.is_file()]
(O / 'SHA256SUMS').write_text('\n'.join(lines) + '\n')
Z = O.with_suffix('.zip')
with zipfile.ZipFile(Z, 'w', zipfile.ZIP_DEFLATED) as z:
    for p in sorted(O.rglob('*')):
        if p.is_file(): z.write(p, f'{NAME}/{p.relative_to(O)}')
print('zip', Z, Z.stat().st_size, h(Z))
