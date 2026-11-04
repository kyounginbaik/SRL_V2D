"""spoon3537: package = the combined final (FINAL_20261104 + ep32 T32c_b1 + ep33 R6M2, sha 8b6fa2b9) with ONLY episode EP's rows rebuilt
from a spoon3537 build-v2 candidate (rules of t33pkg.py / team k33splice: achieved poses of env 0 from results/episode_X.parquet,
provenance columns from the base rows, checkpoint_sha256 = the candidate's checkpoint; step-0 mesh cell kept from the base when the
candidate mesh is byte-identical to the v29 selection mesh, else kit budget_mesh + encode). Verifies that only EP rows / pose+checkpoint
(+mesh) columns differ, then the kit OFFLINE validation (validate_submission + CD-O). No Kaggle login, nothing uploaded.
usage: s3pkg.py EP CAND_TAG LABEL [BASE_PARQUET]"""
import sys, json, shutil, hashlib, importlib.util, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, '/data1/SRL_V2D/challenge_submissions_SRL/analysis_20261020/scripts')
import public_kit_score as K
EP = int(sys.argv[1]); tag, lab = sys.argv[2], sys.argv[3]
BASEP = Path(sys.argv[4]) if len(sys.argv) > 4 else Path('/data1/hansol/teacher33/packages/track3_FINAL_ep32T32cb1_ep33R6M2_20261105/all_20_episodes/track3_submission.parquet')
KITSRC = Path('/data1/hansol/teacher33/packages/track3_FINAL_ep32T32cb1_ep33R6M2_20261105/all_20_episodes/submission_kit')
C22 = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028'); CANDS = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates')
NM = f'episode_{EP:06d}'; POSE = ['pos_x', 'pos_y', 'pos_z', 'quat_x', 'quat_y', 'quat_z', 'quat_w']
REFC = CANDS / json.loads((C22 / 'selection_v29.json').read_text())[str(EP)] / NM
CAND = Path('/data1/hansol/spoon3537/rt/candidates') / tag / NM; OUT = Path('/data1/hansol/spoon3537/rt/packages') / lab / 'all_20_episodes'
assert not OUT.exists(), OUT
h = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
comp = json.loads((CAND / 'completed.json').read_text()); assert comp.get('exact_previous_physical_states_actions') is True
ck = comp['checkpoint']['sha256'] if isinstance(comp.get('checkpoint'), dict) else None; assert ck
base = pd.read_parquet(BASEP)
r = pd.read_parquet(CAND / f'results/{NM}.parquet'); r = r[r.environment_id == r.environment_id.min()]
names = r.drop_duplicates('object_id').sort_values('object_id')[['object_id', 'object_body_name']].values.tolist()
m = base.row_id.str.startswith(f'track_3/e{EP:06d}/'); e = base[m].set_index('row_id')
prov = [c for c in e.columns if c not in POSE + ['object_mesh_b64']]
mesh = {}
for oid, name in names:
    a = next((CAND / 'meshes').rglob(f'{name}.obj')); b = next((REFC / 'meshes').rglob(f'{name}.obj'))
    rid0 = f'track_3/e{EP:06d}/w000/f000000/o{int(oid)}'
    if h(a) == h(b): mesh[int(oid)] = (e.loc[rid0, 'object_mesh_b64'], 'base cell (mesh byte-identical to the v29 selection)')
    else:
        v, f = K.budget_mesh(str(a), 4096, 4096); mesh[int(oid)] = (K.encode(vertices=np.asarray(v, np.float64), faces=np.asarray(f, np.float64)), 'encoded from our mesh')
out = []
for _, x in r.sort_values(['step', 'object_id']).iterrows():
    rid = f'track_3/e{EP:06d}/w000/f{int(x.step):06d}/o{int(x.object_id)}'; t = e.loc[rid]
    row = dict(row_id=rid, pos_x=float(x.achieved_position_x), pos_y=float(x.achieved_position_y), pos_z=float(x.achieved_position_z),
               quat_x=float(x.achieved_quaternion_x), quat_y=float(x.achieved_quaternion_y), quat_z=float(x.achieved_quaternion_z), quat_w=float(x.achieved_quaternion_w),
               **{c: t[c] for c in prov}, object_mesh_b64=mesh[int(x.object_id)][0] if int(x.step) == 0 else '')
    row['checkpoint_sha256'] = ck; out.append(row)
assert len(out) == int(m.sum()), (len(out), int(m.sum()))
new = pd.DataFrame(out)[base.columns]
s = pd.concat([base[~m], new]).set_index('row_id').loc[base.row_id].reset_index()[base.columns]
OUT.mkdir(parents=True); P = OUT / 'track3_submission.parquet'; s.to_parquet(P, index=False)
shutil.copytree(KITSRC, OUT / 'submission_kit', ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.receipt.json'))
a2 = pd.read_parquet(P).set_index('row_id'); b2 = base.set_index('row_id')
d = (a2 != b2) & ~(a2.isna() & b2.isna()); rr = d.any(axis=1)
eps = sorted(set(i.split('/')[1] for i in a2.index[rr])); cols = list(d.columns[d.any()])
assert eps in ([], [f'e{EP:06d}']), eps; assert set(cols) <= set(POSE + ['checkpoint_sha256', 'object_mesh_b64']), cols
print('VERIFY rows differ', int(rr.sum()), 'episodes', eps, 'columns', cols)
if rr.any():
    dp = np.abs(a2.loc[rr, ['pos_x', 'pos_y', 'pos_z']].to_numpy(float) - b2.loc[rr, ['pos_x', 'pos_y', 'pos_z']].to_numpy(float)).max()
    print('max |pos diff| vs base (m)', dp)
pv = [c for c in s.columns if c.endswith('_sha256') and c != 'checkpoint_sha256'] + ['code_commit_url']
print('provenance uniform:', {c: int(s[c].nunique()) for c in pv if c in s.columns})
kit = OUT / 'submission_kit'; sys.path[:0] = [str(kit), str(kit / 'submit')]
spec = importlib.util.spec_from_file_location('v2d_submit', kit / 'submit/v2d_submit.py'); S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
cfg = json.loads(Path(S.CONFIG).read_text()); fr = S.validate_submission(P, 'track_3', cfg)
print('KIT validate_submission OK:', len(fr), 'rows,', fr[S.rs.ROW_ID_COLUMN].astype(str).str.split('/').str[1].nunique(), 'episodes')
from v2dlb.track3_pose import cd_o_submission
smp = kit / 'data/track_3_mesh_sample_submission.parquet'; obj = kit / 'data/track_3_evaluation_objects.json'
cdo = P.with_name(P.stem + '.cd-o.parquet'); cd_o_submission(fr, pd.read_parquet(smp), json.loads(obj.read_text())['episodes']).to_parquet(cdo, index=False)
S.validate_submission(cdo, 'track_3_mesh', cfg); print('KIT CD-O derivation + validation OK:', cdo.name)
info = dict(label=lab, episode=EP, base=str(BASEP), base_sha=h(BASEP), candidate=str(CAND), checkpoint_sha256=ck, rows_replaced=int(m.sum()),
            mesh={k: v[1] for k, v in mesh.items()}, columns_changed=cols, sha=h(P), cdo_sha=h(cdo), uploaded=False)
(OUT.parent / 'S3_PACKAGE.json').write_text(json.dumps(info, indent=1)); print('SHA', info['sha'], 'CDO', info['cdo_sha'][:16])
