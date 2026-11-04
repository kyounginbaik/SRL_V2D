"""R33: full Track 3 package = FINAL_20261104 (team best, sha 0fca055b) with ONLY the ep33 rows rebuilt from our build-v2 candidate
# [T33 copy of retarget33 r33pkg.py: candidates and packages under /data1/hansol/teacher33]
(rules of the team's k33splice.py / v32build.py: achieved poses of env 0 from results/episode_000033.parquet, provenance columns from the
base ep33 rows, checkpoint_sha256 = the candidate's checkpoint; step-0 mesh cell kept from the base when our mesh file is byte-identical to
the base ep33 candidate's (K33_H5r16_c3), else the kit budget_mesh+encode of our mesh). Then: verify only ep33 rows / pose+checkpoint(+mesh)
columns differ, and run the kit's OFFLINE validation (validate_submission + CD-O derivation). No Kaggle login, nothing uploaded.
usage: r33pkg.py CAND_TAG LABEL"""
import sys, json, shutil, hashlib, importlib.util, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, '/data1/SRL_V2D/challenge_submissions_SRL/analysis_20261020/scripts')
import public_kit_score as K
BASE = Path('/data1/SRL_V2D/challenge_submissions_SRL/FINAL_20261104/all_20_episodes')
REFC = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates/K33_K33_H5r16_c3/episode_000033')
EP = 33; NM = f'episode_{EP:06d}'; POSE = ['pos_x', 'pos_y', 'pos_z', 'quat_x', 'quat_y', 'quat_z', 'quat_w']
tag, lab = sys.argv[1], sys.argv[2]
CAND = Path('/data1/hansol/teacher33/candidates') / tag / NM; OUT = Path('/data1/hansol/teacher33/packages') / lab / 'all_20_episodes'
assert not OUT.exists(), OUT
h = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
comp = json.loads((CAND / 'completed.json').read_text()); assert comp.get('exact_previous_physical_states_actions') is True
ck = comp['checkpoint']['sha256'] if isinstance(comp.get('checkpoint'), dict) else None; assert ck
base = pd.read_parquet(BASE / 'track3_submission.parquet')
r = pd.read_parquet(CAND / f'results/{NM}.parquet'); r = r[r.environment_id == r.environment_id.min()]
names = r.drop_duplicates('object_id').sort_values('object_id')[['object_id', 'object_body_name']].values.tolist()
m = base.row_id.str.startswith(f'track_3/e{EP:06d}/'); e = base[m].set_index('row_id')
prov = [c for c in e.columns if c not in POSE + ['object_mesh_b64']]
mesh = {}
for oid, name in names:
    a = next((CAND / 'meshes').rglob(f'{name}.obj')); b = next((REFC / 'meshes').rglob(f'{name}.obj'))
    rid0 = f'track_3/e{EP:06d}/w000/f000000/o{int(oid)}'
    if h(a) == h(b): mesh[int(oid)] = (e.loc[rid0, 'object_mesh_b64'], 'base cell (mesh byte-identical to K33_H5r16_c3)')
    else:
        v, f = K.budget_mesh(str(a), 4096, 4096); mesh[int(oid)] = (K.encode(vertices=np.asarray(v, np.float64), faces=np.asarray(f, np.float64)), 'encoded from our mesh')
out = []
for _, x in r.sort_values(['step', 'object_id']).iterrows():
    rid = f'track_3/e{EP:06d}/w000/f{int(x.step):06d}/o{int(x.object_id)}'; t = e.loc[rid]
    row = dict(row_id=rid, pos_x=float(x.achieved_position_x), pos_y=float(x.achieved_position_y), pos_z=float(x.achieved_position_z),
               quat_x=float(x.achieved_quaternion_x), quat_y=float(x.achieved_quaternion_y), quat_z=float(x.achieved_quaternion_z), quat_w=float(x.achieved_quaternion_w),
               **{c: t[c] for c in prov}, object_mesh_b64=mesh[int(x.object_id)][0] if int(x.step) == 0 else '')
    row['checkpoint_sha256'] = ck; out.append(row)
assert len(out) == int(m.sum()) == 696, (len(out), int(m.sum()))
new = pd.DataFrame(out)[base.columns]
s = pd.concat([base[~m], new]).set_index('row_id').loc[base.row_id].reset_index()[base.columns]
OUT.mkdir(parents=True); P = OUT / 'track3_submission.parquet'; s.to_parquet(P, index=False)
shutil.copytree(BASE / 'submission_kit', OUT / 'submission_kit', ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.receipt.json'))
(OUT / 'results').mkdir()
for p in sorted((BASE / 'results').glob('episode_*.parquet')):
    shutil.copy2(CAND / f'results/{NM}.parquet' if p.name == f'{NM}.parquet' else p, OUT / 'results' / p.name)
# verify
a2 = pd.read_parquet(P).set_index('row_id'); b2 = base.set_index('row_id')
d = (a2 != b2) & ~(a2.isna() & b2.isna()); rr = d.any(axis=1)
eps = sorted(set(i.split('/')[1] for i in a2.index[rr])); cols = list(d.columns[d.any()])
assert eps == ['e000033'], eps; assert set(cols) <= set(POSE + ['checkpoint_sha256', 'object_mesh_b64']), cols
print('VERIFY rows differ', int(rr.sum()), 'episodes', eps, 'columns', cols)
# kit offline validation (same functions the kit's v2d_submit.py calls before any Kaggle call)
kit = OUT / 'submission_kit'; sys.path[:0] = [str(kit), str(kit / 'submit')]
spec = importlib.util.spec_from_file_location('v2d_submit', kit / 'submit/v2d_submit.py'); S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
cfg = json.loads(Path(S.CONFIG).read_text()); fr = S.validate_submission(P, 'track_3', cfg)
print('KIT validate_submission OK:', len(fr), 'rows,', fr[S.rs.ROW_ID_COLUMN].astype(str).str.split('/').str[1].nunique(), 'episodes')
from v2dlb.track3_pose import cd_o_submission
smp = kit / 'data/track_3_mesh_sample_submission.parquet'; obj = kit / 'data/track_3_evaluation_objects.json'
cdo = P.with_name(P.stem + '.cd-o.parquet'); cd_o_submission(fr, pd.read_parquet(smp), json.loads(obj.read_text())['episodes']).to_parquet(cdo, index=False)
S.validate_submission(cdo, 'track_3_mesh', cfg); print('KIT CD-O derivation + validation OK:', cdo.name)
info = dict(label=lab, base=str(BASE), base_sha=h(BASE / 'track3_submission.parquet'), candidate=str(CAND), checkpoint_sha256=ck,
            rows_replaced=int(m.sum()), mesh={k: v[1] for k, v in mesh.items()}, columns_changed=cols, sha=h(P), cdo_sha=h(cdo),
            uploaded=False, note='ep33 only; candidate ' + tag + ' (target = the TG part of the tag, our GT-free recon); build-v2 exact')
(OUT.parent / 'R33_PACKAGE.json').write_text(json.dumps(info, indent=1)); print('SHA', info['sha'], 'CDO', info['cdo_sha'][:16])
