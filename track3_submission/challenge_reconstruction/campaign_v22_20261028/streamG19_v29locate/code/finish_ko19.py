"""G19: assemble a v29-based knockout/swap package with the unchanged v29 packer chain (campaign_v22_20261028/v29/finish_v29.py logic:
assemble_twenty_v2.py -> make_srl_folder.py). The selection file is v29's with only the listed suspect episodes swapped to other
build-v2 components (exact replay + portable archive required, as in v29). Videos: kept episodes reuse v29's video dirs (same
rollout check); swapped episodes get no review video (diagnostic MP-SR upload). Does not upload.
usage: finish_ko19.py SELECTION.json OUT_DIR   (OUT_DIR must not exist; writes OUT_DIR/build and OUT_DIR/all_20_episodes)"""
import json, os, sys, hashlib, subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/v29')
import finish_v29 as F  # reuse read/sha/same_rollout/run/inventory/package_args and constants (no side effects at import)

SEL, OUTD = Path(sys.argv[1]), Path(sys.argv[2])
if OUTD.exists(): raise SystemExit(f'refusing to overwrite {OUTD}')
V29SEL = F.read(F.C / 'selection_v29.json'); SELJ = F.read(SEL)
assert sorted(SELJ) == sorted(V29SEL), 'selection must cover the same 20 episodes'
SWAP = sorted(int(k) for k in SELJ if SELJ[k] != V29SEL[k])
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[key] = '1'


def comp_dir(tag, name):
    return (F.CAND / tag / name) if (F.CAND / tag / name).is_dir() else (F.CAND_OVERFLOW / tag / name)


chosen = {}
for key, tag in SELJ.items():
    ep = int(key); name = f'episode_{ep:06d}'; out = comp_dir(tag, name); done = F.read(out / 'completed.json')
    if done.get('exact_previous_physical_states_actions') is not True: raise ValueError(f'{tag}: build-v2 replay not exact')
    legacy = Path(done['legacy_component']); portable = legacy.parents[1] / 'portable_archives' / f'{name}_portable.tar.gz'
    if not portable.is_file(): raise FileNotFoundError(portable)
    chosen[ep] = dict(tag=tag, component=str(out), portable=str(portable), legacy=str(legacy), swapped=ep in SWAP)
args = []
for ep, item in sorted(chosen.items()):
    args += ['--component', f'{ep}={item["component"]}', '--portable', f'{ep}={item["portable"]}']
    if item['swapped']: continue
    for v in ('v29', 'v28', 'v27', 'v26', 'v25', 'v24', 'v23b', 'v23'):
        vd = F.C / v / 'videos' / f'episode_{ep:06d}'
        if (vd / 'render_report.json').is_file(): break
    else:
        vd = F.C / 'videos_v22' / f'episode_{ep:06d}'
    report = F.read(vd / 'render_report.json'); execution = F.read(Path(item['component']) / 'backend/evaluation/result.json')['execution']
    if not F.same_rollout(report['refined_trajectory'], execution['trajectory']): raise ValueError(f'video rollout differs: {ep}')
    args += ['--video', f'{ep}={vd}']
OUT = OUTD / 'build'; SRL = OUTD / 'all_20_episodes'; LOG = OUTD / 'logs'; LOG.mkdir(parents=True)
(OUTD / 'selection.json').write_text(json.dumps(SELJ, indent=1) + '\n')
F.run(F.C16 / 'package_code/assemble_twenty_v2.py', '--build-manifest', F.BUILD / 'build_manifest.json',
      '--common-evaluator', F.BUILD / 'package_code/common_evaluator.py', *args, '--output', OUT,
      '--reason', f'G19 diagnostic knockout/swap package: v29 with episodes {SWAP} swapped to other build-v2 policy rollouts '
      f'({", ".join(f"{e}={SELJ[str(e)]}" for e in SWAP)}). All components reproduce exactly under evaluator build v2. Our meshes only.',
      log=LOG / 'assemble.log')
note = OUTD / 'note.md'
note.write_text(f'\n## G19 knockout/swap package (2026-11-01)\n\nv29 with episodes {SWAP} swapped: ' + '; '.join(f'ep{e} {V29SEL[str(e)]} -> {SELJ[str(e)]}' for e in SWAP)
                + '. Diagnostic MP-SR upload to locate the v29 failures; not a leaderboard candidate. Nothing uploaded.\n')
F.inventory(OUT)
F.run(F.ROOT / 'challenge_submissions_SRL/make_srl_folder.py', OUT, SRL, note, log=LOG / 'srl.log')
if F.sha(OUT / 'track3_submission.parquet') != F.sha(SRL / 'track3_submission.parquet'): raise ValueError('SRL parquet differs')
F.inventory(SRL)
# row-level diff vs v29: only swapped episodes may differ
import pandas as pd, numpy as np
a = pd.read_parquet(F.SRL / 'track3_submission.parquet'); b = pd.read_parquet(SRL / 'track3_submission.parquet')
epcol = 'row_id[8:15]'
a['_e'] = a.row_id.str.slice(9, 15).astype(int); b['_e'] = b.row_id.str.slice(9, 15).astype(int)
diff = {}
for ep in sorted(set(a['_e'])):
    x = a[a['_e'] == ep].reset_index(drop=True); y = b[b['_e'] == ep].reset_index(drop=True)
    same = x.shape == y.shape and all((x[c].astype(str) == y[c].astype(str)).all() for c in x.columns)
    diff[str(ep)] = not same
rep = dict(selection=SELJ, swapped=SWAP, sha256=F.sha(SRL / 'track3_submission.parquet'), episode_column=epcol,
           episodes_differing_from_v29=[e for e, d in diff.items() if d], v29_sha256=F.sha(F.SRL / 'track3_submission.parquet'))
(OUTD / 'build_report.json').write_text(json.dumps(rep, indent=1) + '\n'); print(json.dumps(rep, indent=1))
