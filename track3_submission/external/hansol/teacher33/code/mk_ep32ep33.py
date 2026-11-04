"""Combine: FINAL_20261104 + ep32 rows from the team's T32c_b1 package (bd5b3321, MP-SR 0.95) + ep33 rows from our R6M2 package
(c96244f0, MP-SR 0.95). Verify vs FINAL that only e000032/e000033 differ and each equals its donor exactly; kit offline validation + CD-O.
No Kaggle login, nothing uploaded. usage: mk_ep32ep33.py LABEL"""
import sys, json, shutil, hashlib, importlib.util, pandas as pd
from pathlib import Path
h = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
FIN = Path('/data1/SRL_V2D/challenge_submissions_SRL/FINAL_20261104/all_20_episodes')
E32 = Path('/data1/SRL_V2D/challenge_submissions_SRL/track3_final_ep32_T32c_b1_20261104/all_20_episodes')
E33 = Path('/data1/hansol/teacher33/packages/track3_FINAL_ep33T33_R6M2Cf0_r6moved_20261104/all_20_episodes')
OUT = Path('/data1/hansol/teacher33/packages') / sys.argv[1] / 'all_20_episodes'; assert not OUT.exists(), OUT
for p, s in ((E32, 'bd5b3321'), (E33, 'c96244f0')): assert h(p / 'track3_submission.parquet').startswith(s), p
f = pd.read_parquet(FIN / 'track3_submission.parquet'); a = pd.read_parquet(E32 / 'track3_submission.parquet'); b = pd.read_parquet(E33 / 'track3_submission.parquet')
assert list(f.columns) == list(a.columns) == list(b.columns) and (f.row_id.values == a.row_id.values).all() and (f.row_id.values == b.row_id.values).all()
m32 = f.row_id.str.contains('/e000032/'); m33 = f.row_id.str.contains('/e000033/'); print('rows ep32', int(m32.sum()), 'ep33', int(m33.sum()))
o = b.copy(); o.loc[m32] = a.loc[m32].values
def diff(x, y):
    d = (x != y) & ~(x.isna() & y.isna()); r = d.any(axis=1); return sorted(set(i.split('/')[1] for i in x.row_id[r])), list(d.columns[d.any()])
for nm, x in (('ep32 pkg', a), ('ep33 pkg', b), ('combined', o)): print('vs FINAL', nm, diff(x, f))
assert diff(o, f)[0] == ['e000032', 'e000033']
assert diff(o[m32], a[m32])[0] == [] and diff(o[m33], b[m33])[0] == [] and diff(o[~(m32 | m33)], f[~(m32 | m33)])[0] == []
prov = [c for c in f.columns if c not in ('row_id', 'pos_x', 'pos_y', 'pos_z', 'quat_x', 'quat_y', 'quat_z', 'quat_w', 'object_mesh_b64', 'checkpoint_sha256')]
print('provenance columns uniform over 20 episodes:', {c: int(o[c].nunique()) for c in prov})
OUT.mkdir(parents=True); P = OUT / 'track3_submission.parquet'; o.to_parquet(P, index=False)
shutil.copytree(E33 / 'submission_kit', OUT / 'submission_kit', ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.receipt.json'))
kit = OUT / 'submission_kit'; sys.path[:0] = [str(kit), str(kit / 'submit')]
spec = importlib.util.spec_from_file_location('v2d_submit', kit / 'submit/v2d_submit.py'); S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
cfg = json.loads(Path(S.CONFIG).read_text()); fr = S.validate_submission(P, 'track_3', cfg)
print('KIT validate_submission OK:', len(fr), 'rows,', fr[S.rs.ROW_ID_COLUMN].astype(str).str.split('/').str[1].nunique(), 'episodes')
from v2dlb.track3_pose import cd_o_submission
smp = kit / 'data/track_3_mesh_sample_submission.parquet'; obj = kit / 'data/track_3_evaluation_objects.json'
cdo = P.with_name(P.stem + '.cd-o.parquet'); cd_o_submission(fr, pd.read_parquet(smp), json.loads(obj.read_text())['episodes']).to_parquet(cdo, index=False)
S.validate_submission(cdo, 'track_3_mesh', cfg); print('KIT CD-O derivation + validation OK')
info = dict(label=sys.argv[1], final=str(FIN), final_sha=h(FIN / 'track3_submission.parquet'), ep32_from=str(E32), ep32_sha=h(E32 / 'track3_submission.parquet'),
            ep33_from=str(E33), ep33_sha=h(E33 / 'track3_submission.parquet'), sha=h(P), cdo_sha=h(cdo), uploaded=False,
            note='FINAL_20261104 mesh cells; ep32 = T32c_b1 (MP-SR 0.95 alone), ep33 = R6M2 (MP-SR 0.95 alone)')
(OUT.parent / 'COMBINED_PACKAGE.json').write_text(json.dumps(info, indent=1)); print('SHA', info['sha']); print('CDO', info['cdo_sha'])
