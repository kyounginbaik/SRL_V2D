"""Write out/package_summary.md for every v30c package."""
import json, hashlib, re
from pathlib import Path
S = Path('/data1/SRL_V2D/challenge_submissions_SRL'); X = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamC30_compose')
logs = ''.join(p.read_text(errors='ignore') for p in X.glob('logs/*.log'))
zips = dict(re.findall(r'(/data1\S+\.zip) \d+ ([0-9a-f]{64})', logs))
dry = X.joinpath('logs/dryrun_all.log').read_text().split('== ')[1:]
dry = {b.split('\n')[0].strip(): ('local validation OK' if 'looks valid' in b else 'local FAIL') + ('; Kaggle preflight: rules-not-accepted (account side, same for v29)' if 'not accepted' in b else ('; preflight OK' if 'nothing submitted' in b else '')) for b in dry}
def agg(d):
    n = len(d); return '%d/%d, MPPE %.3f, AUC %.4f%s' % (sum(v['scores']['MP-SR'] for v in d.values()), n, sum(v['scores']['MPPE'] for v in d.values()) / n,
           sum(v['scores']['AUC'] for v in d.values()) / n, (' fail ' + ','.join(e for e, v in d.items() if v['scores']['MP-SR'] != 1)) if any(v['scores']['MP-SR'] != 1 for v in d.values()) else '')
rows = ['| package | parquet sha256 | zip sha256 | vs own target (MP, MPPE, AUC) | vs target_final_v14_g17 | dry-run |', '|---|---|---|---|---|---|']
for k in range(8, 0, -1):
    for v in ('', '_cwmesh', '_cwmesh_nogt'):
        name = f'track3_v30c{k}{v}_20261101'; P = S / name / 'all_20_episodes'
        ps = hashlib.sha256((P / 'track3_submission.parquet').read_bytes()).hexdigest()
        if v == '':
            own = agg(json.loads((P / f'target_compose_v{k}_on_v14_scores.json').read_text())); v14 = agg(json.loads((P / 'target_final_v14_g17_scores.json').read_text()))
        else:
            f1 = X / f'out/pkgscores/v30c{k}{v}_vs_c{k}.json'; f2 = X / f'out/pkgscores/v30c{k}{v}_vs_v14.json'
            own = agg(json.loads(f1.read_text())['episodes']) if f1.exists() else 'not scored'
            v14 = agg(json.loads(f2.read_text())['episodes']) if f2.exists() else 'not scored (same parquet as v30c6' + v + ')' if k < 6 else 'not scored'
        rows.append(f"| {name} | {ps[:12]} | {zips.get(str(S / (name + '.zip')), 'n/a')[:12]} | {own} | {v14} | {dry.get(name, 'n/a')} |")
(X / 'out/package_summary.md').write_text('\n'.join(rows) + '\n'); print('\n'.join(rows))
