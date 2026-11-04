"""K35 DIAGNOSTIC single-episode package: v29 parquet with the ep35 rows rebuilt from a build-v2 candidate (achieved poses of env 0 from
results/episode_000035.parquet, step-0 mesh payload = kit budget_mesh+encode of the candidate's own mesh, provenance columns = v29's ep35
rows with checkpoint_sha256 = the candidate's checkpoint). --check: rebuild v29's own ep35 rows from its v29 candidate and assert equality.
usage: k35splice.py CAND_TAG OUT_PKG_DIR LABEL | --check"""
import sys, json, shutil, hashlib, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, '/data1/SRL_V2D/challenge_submissions_SRL/analysis_20261020/scripts')
import public_kit_score as K
CAND = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates')
V29 = Path('/data1/SRL_V2D/challenge_submissions_SRL/track3_v29_20261030/all_20_episodes')
EP = 35; NM = f'episode_{EP:06d}'
def rows_for(tag, base):
    tagd = CAND / tag / NM; comp = json.loads((tagd / 'completed.json').read_text())
    r = pd.read_parquet(tagd / f'results/{NM}.parquet'); r = r[r.environment_id == r.environment_id.min()]
    names = r.drop_duplicates('object_id').sort_values('object_id')[['object_id', 'object_body_name']].values.tolist(); pay = {}
    for oid, name in names:
        m = next((tagd / 'meshes').rglob(f'{name}.obj')); v, f = K.budget_mesh(str(m), 4096, 4096)
        pay[int(oid)] = K.encode(vertices=np.asarray(v, np.float64), faces=np.asarray(f, np.float64))
    e = base[base.row_id.str.startswith(f'track_3/e{EP:06d}/')].set_index('row_id')
    prov = [c for c in e.columns if c not in ('pos_x', 'pos_y', 'pos_z', 'quat_x', 'quat_y', 'quat_z', 'quat_w', 'object_mesh_b64')]
    ck = comp.get('checkpoint', {}).get('sha256') if isinstance(comp.get('checkpoint'), dict) else None
    out = []
    for _, x in r.sort_values(['step', 'object_id']).iterrows():
        rid = f"track_3/e{EP:06d}/w000/f{int(x.step):06d}/o{int(x.object_id)}"; t = e.loc[rid]
        row = dict(row_id=rid, pos_x=float(x.achieved_position_x), pos_y=float(x.achieved_position_y), pos_z=float(x.achieved_position_z),
                   quat_x=float(x.achieved_quaternion_x), quat_y=float(x.achieved_quaternion_y), quat_z=float(x.achieved_quaternion_z),
                   quat_w=float(x.achieved_quaternion_w), **{c: t[c] for c in prov}, object_mesh_b64=pay[int(x.object_id)] if int(x.step) == 0 else '')
        if ck: row['checkpoint_sha256'] = ck
        out.append(row)
    assert len(out) == len(e)
    df = pd.DataFrame(out)[base.columns]
    # mesh payload: if the candidate's mesh file is byte-identical to the v29 component's, reuse v29's exact step-0 cell (the kit encoder's
    # zip bytes are not deterministic across runs; the decoded arrays are the same)
    v29tag = json.loads(Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/selection_v29.json').read_text())[str(EP)]
    for oid, name in names:
        a = next((tagd / 'meshes').rglob(f'{name}.obj')); b = next((CAND / v29tag / NM / 'meshes').rglob(f'{name}.obj'))
        if hashlib.sha256(a.read_bytes()).digest() == hashlib.sha256(b.read_bytes()).digest():
            rid = f"track_3/e{EP:06d}/w000/f000000/o{int(oid)}"; df.loc[df.row_id == rid, 'object_mesh_b64'] = e.loc[rid, 'object_mesh_b64']
    return df, ck, comp
base = pd.read_parquet(V29 / 'track3_submission.parquet')
if sys.argv[1] == '--check':
    new, ck, comp = rows_for(json.loads(Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/selection_v29.json').read_text())['35'], base)
    old = base[base.row_id.str.startswith(f'track_3/e{EP:06d}/')].reset_index(drop=True)
    cols_diff = [c for c in base.columns if not (new[c].reset_index(drop=True).equals(old[c]))]
    print('checkpoint from completed.json', ck, '| v29 ep35 checkpoint', old.checkpoint_sha256.iloc[0], '| differing columns', cols_diff)
    for c in cols_diff[:3]:
        a, b = new[c].reset_index(drop=True), old[c]; i = (a != b).idxmax(); print(c, repr(a[i])[:80], repr(b[i])[:80])
    sys.exit()
tag, out, lab = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
assert not out.exists(); out.mkdir(parents=True)
new, ck, comp = rows_for(tag, base)
s = base.copy(); m = s.row_id.str.startswith(f'track_3/e{EP:06d}/')
s = pd.concat([s[~m], new]).set_index('row_id').loc[base.row_id].reset_index()
s.to_parquet(out / 'track3_submission.parquet', index=False)
shutil.copytree(V29 / 'submission_kit', out / 'submission_kit', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
(out / 'results').mkdir(); [shutil.copy2(p, out / 'results' / p.name) for p in (V29 / 'results').glob('*.parquet') if p.name != f'{NM}.parquet']
shutil.copy2(CAND / tag / NM / f'results/{NM}.parquet', out / f'results/{NM}.parquet')
h = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
info = dict(diagnostic_only=True, label=lab, episode=EP, base='track3_v29_20261030', base_sha=h(V29 / 'track3_submission.parquet'), candidate=tag,
            candidate_checkpoint_sha256=ck, sha=h(out / 'track3_submission.parquet'), rows_replaced=int(m.sum()))
(out.parent / 'K35_PACKAGE.json').write_text(json.dumps(info, indent=1))
(out.parent / 'README.md').write_text(f'# DIAGNOSTIC K35 ep35 swap {lab}\n\nv29 with only episode 35 rows rebuilt from build-v2 candidate `{tag}` (K35 hypothesis retarget).\n'
    'Upload with --only maniptrans_sr: MP 0.90 -> this ep35 passes officially; 0.85 -> it still fails.\n')
print(lab, info['sha'][:8], info['rows_replaced'], ck)
