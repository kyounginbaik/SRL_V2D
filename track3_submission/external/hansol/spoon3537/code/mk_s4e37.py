"""8b6fa2b9 + ep36 s4 (9bc322af) + ep37 s1a: verify only e000036/e000037 differ and equal their donors; kit validate + CD-O."""
import sys, json, shutil, hashlib, importlib.util, pandas as pd
from pathlib import Path
h = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
P = Path('/data1/hansol/spoon3537/rt/packages')
BASE = Path('/data1/hansol/teacher33/packages/track3_FINAL_ep32T32cb1_ep33R6M2_20261105/all_20_episodes')
S4 = P / 'track3_FINALc_ep36s4/all_20_episodes'; E37 = P / 'track3_FINALc_ep37s1a_tgt_s1/all_20_episodes'
OUT = P / 'track3_FINALc_B_ep36s4_ep37s1a/all_20_episodes'; assert not OUT.exists()
b = pd.read_parquet(BASE / 'track3_submission.parquet'); a = pd.read_parquet(S4 / 'track3_submission.parquet'); c = pd.read_parquet(E37 / 'track3_submission.parquet')
print('base', h(BASE / 'track3_submission.parquet')[:8], 's4', h(S4 / 'track3_submission.parquet')[:8], 'e37', h(E37 / 'track3_submission.parquet')[:8])
assert (b.row_id.values == a.row_id.values).all() and (b.row_id.values == c.row_id.values).all()
m36 = b.row_id.str.contains('/e000036/'); m37 = b.row_id.str.contains('/e000037/')
def diff(x, y):
    d = (x != y) & ~(x.isna() & y.isna()); r = d.any(axis=1); return sorted(set(i.split('/')[1] for i in x.row_id[r]))
print('s4 vs base', diff(a, b), 'e37 vs base', diff(c, b))
o = a.copy(); o.loc[m37] = c.loc[m37].values
assert diff(o, b) == ['e000036', 'e000037'] and diff(o[m36], a[m36]) == [] and diff(o[m37], c[m37]) == []
prov = ['eval_script_sha256', 'eval_code_sha256', 'packer_script_sha256', 'uploader_script_sha256', 'code_commit_url']
print('provenance uniform', {k: int(o[k].nunique()) for k in prov})
OUT.mkdir(parents=True); F = OUT / 'track3_submission.parquet'; o.to_parquet(F, index=False)
shutil.copytree(BASE / 'submission_kit', OUT / 'submission_kit', ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.receipt.json'))
kit = OUT / 'submission_kit'; sys.path[:0] = [str(kit), str(kit / 'submit')]
spec = importlib.util.spec_from_file_location('v2d_submit', kit / 'submit/v2d_submit.py'); S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
cfg = json.loads(Path(S.CONFIG).read_text()); fr = S.validate_submission(F, 'track_3', cfg); print('KIT OK', len(fr))
from v2dlb.track3_pose import cd_o_submission
cdo = F.with_name('track3_submission.cd-o.parquet'); cd_o_submission(fr, pd.read_parquet(kit / 'data/track_3_mesh_sample_submission.parquet'), json.loads((kit / 'data/track_3_evaluation_objects.json').read_text())['episodes']).to_parquet(cdo, index=False)
S.validate_submission(cdo, 'track_3_mesh', cfg); print('CDO OK'); print('SHA', h(F))
