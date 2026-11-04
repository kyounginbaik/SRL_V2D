"""C30 coworker-mesh package (mesh-only swap at package level; poses, results, robot_actions untouched).
SRC = an assembled SRL package dir (.../track3_v30cK_20261101/all_20_episodes), DST = new dir (never overwrites).
VARIANT cw   : coworker final_episode_meshes_20261029 meshes (rigidly expressed in our body frame by cwalign.py) for every non-spoon object
               that passes the kit-registration twin check (cups: rotation only about their symmetry axis is accepted);
               wooden_piece_1/2 fail it (180 deg twin flips across kit seeds) -> our meshes kept.
VARIANT nogt : as cw but our meshes for plastic_dish_rack, white_pot_lid, water_pitcher (objects the coworker README marks
               proxy-/GT-guided now or in earlier versions).
Spoon: no coworker spoon -> G18 spoon kept. Payload = kit budget_mesh(4096/4096) + kit encode, as in make_sub_any.py.
usage: cwpkg.py SRC DST VARIANT"""
import sys, json, shutil, hashlib, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, '/data1/SRL_V2D/challenge_submissions_SRL/analysis_20261020/scripts')
import public_kit_score as K
X = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamC30_compose')
SRC, DST, VAR = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
assert VAR in ('cw', 'nogt') and SRC.is_dir() and not DST.exists() and not str(DST.resolve()).startswith(str(SRC.resolve()))
rep = {(r['ep'], r['obj']): r for r in json.loads((X / 'cwmesh/cwalign_report.json').read_text())}
SYM_OK = json.loads((X / 'cwmesh/cup_symcheck.json').read_text())          # {"ep:obj": true} cups whose kit rotations are about the symmetry axis
GT_LABEL = {'plastic_dish_rack': 'proxy-guided (coworker README: stereo Poisson deformed toward a hand-specified target, "no GT geometry"; the earlier 10-29 rack was GT-guided)',
            'white_pot_lid': 'proxy-guided (coworker README: data-only lid deformed toward a hand-specified cone+lip+handle target; earlier lid was GT-guided / remeshed public lid)',
            'water_pitcher': 'data-only per coworker README 2026-10-30 (earlier compose versions used the public scan / a GT-guided pitcher)'}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
shutil.copytree(SRC, DST, ignore=shutil.ignore_patterns('*.kaggle-receipt.json', 'track3_submission.cd-o.parquet', '__pycache__', '*.pyc'))
sub = pd.read_parquet(DST / 'track3_submission.parquet'); orig = sub.copy()
swaps, kept = [], []
for (e, n), r in sorted(rep.items()):
    ok = r['kit_pass'] or SYM_OK.get(f'{e}:{n}', False)
    reason = None
    if not ok: reason = 'kit registration twin check failed (180 deg flips across seeds)'
    elif VAR == 'nogt' and n in GT_LABEL: reason = 'nogt variant: guided mesh excluded'
    if reason: kept.append(dict(ep=e, obj=n, reason=reason)); continue
    res = pd.read_parquet(DST / f'results/episode_{e:06d}.parquet', columns=['object_id', 'object_body_name']).drop_duplicates()
    oid = int(res[res.object_body_name == n].object_id.iloc[0])
    v, f = K.budget_mesh(r['out'], 4096, 4096); payload = K.encode(vertices=np.asarray(v, np.float64), faces=np.asarray(f, np.float64))
    rid0 = f'track_3/e{e:06d}/w000/f000000/o{oid}'
    m = sub.row_id == rid0; assert m.sum() == 1, rid0
    sub.loc[m, 'object_mesh_b64'] = payload
    mp = DST / 'meshes' / f'episode_{e:06d}' / f'{n}.obj'; old = sha(mp); mp.unlink(); shutil.copyfile(r['out'], mp)
    swaps.append(dict(ep=e, obj=n, object_id=oid, source=r['src'], aligned=r['out'], rigid_R=r['R'], rigid_t=r['t'], icp_cost_mm=r['icp_cost_mm'],
                      kit_max_rot_deg=r['kit_max_rot_deg'], kit_pass_or_symmetric=True, budget=[len(v), len(f)], old_mesh_sha256=old, new_mesh_sha256=sha(mp),
                      provenance=GT_LABEL.get(n, 'data-only per coworker README (final_episode_meshes_20261029)')))
sub.to_parquet(DST / 'track3_submission.parquet', index=False)
# verification: only the intended mesh payload cells differ
chk = pd.read_parquet(DST / 'track3_submission.parquet')
assert list(chk.columns) == list(orig.columns) and (chk.row_id == orig.row_id).all()
diff = [(c, int((chk[c] != orig[c]).sum())) for c in chk.columns if not chk[c].equals(orig[c])]
assert [c for c, _ in diff] in ([], ['object_mesh_b64']), diff
changed_rows = sorted(chk.row_id[chk.object_mesh_b64 != orig.object_mesh_b64])
want = sorted(f"track_3/e{s['ep']:06d}/w000/f000000/o{s['object_id']}" for s in swaps)
assert changed_rows == want, (changed_rows, want)
info = dict(variant=VAR, base_package=str(SRC), base_parquet_sha256=sha(SRC / 'track3_submission.parquet'),
            parquet_sha256=sha(DST / 'track3_submission.parquet'), changed_rows=changed_rows, swaps=swaps, kept_ours=kept,
            spoon='G18 spoon kept (no coworker spoon mesh)', poses_results_robot_actions='byte-identical to the base package (mesh-only swap)',
            ground_truth_note='Coworker meshes are their own reconstruction; see provenance per swap. Not verified by us beyond their README.')
(DST / 'CWMESH_SWAP.json').write_text(json.dumps(info, indent=1))
lab = {'cw': 'COWORKER MESHES (all non-spoon objects that pass the twin check, incl. rack / lid / pitcher which the coworker marks proxy-guided or previously GT-guided)',
       'nogt': 'COWORKER MESHES for data-only objects only; OUR meshes for plastic_dish_rack, white_pot_lid, water_pitcher'}[VAR]
note = (f'\n## Mesh variant: {lab}\n\nMesh-only swap of {SRC.parent.name} (poses, results, robot actions unchanged; only the step-0 object_mesh_b64 cells of '
        f'{len(swaps)} objects differ). Kept ours: ' + ', '.join(f"ep{k['ep']} {k['obj']} ({k['reason']})" for k in kept) + '. Spoon = G18 (ours).\n\n| ep | object | provenance (coworker README) | kit twin check max rot (deg) |\n|---|---|---|---|\n'
        + ''.join(f"| {s['ep']} | {s['obj']} | {s['provenance']} | {s['kit_max_rot_deg']:.1f}{' (about symmetry axis)' if s['kit_max_rot_deg'] >= 5 else ''} |\n" for s in swaps))
with (DST / 'README.md').open('a') as h: h.write(note)
vals = {str(p.relative_to(DST)): sha(p) for p in sorted(DST.rglob('*')) if p.is_file() and p.name != 'SHA256SUMS.json'}
(DST / 'SHA256SUMS.json').write_text(json.dumps(vals, indent=2) + '\n')
print(json.dumps(dict(parquet_sha256=info['parquet_sha256'], n_swaps=len(swaps), kept=kept, changed_rows=len(changed_rows)), indent=1))
