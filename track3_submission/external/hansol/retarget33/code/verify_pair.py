"""E31: parquet diff of two Track 3 packages: row ids, non-mesh columns, which (episode, object) mesh payloads differ.
usage: verify_pair.py PKG_A(all_20_episodes) PKG_B OUT.json"""
import sys, json, hashlib
import pandas as pd
A, B, OUT = sys.argv[1:4]
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
a, b = pd.read_parquet(f'{A}/track3_submission.parquet'), pd.read_parquet(f'{B}/track3_submission.parquet')
out = dict(a=A, b=B, sha_a=sha(f'{A}/track3_submission.parquet'), sha_b=sha(f'{B}/track3_submission.parquet'), rows=(len(a), len(b)),
           row_ids_equal=bool(len(a) == len(b) and (a.row_id.values == b.row_id.values).all()))
cols = [c for c in a.columns if c != 'object_mesh_b64']
out['non_mesh_columns_differ'] = [c for c in cols if not a[c].equals(b[c])]
k = a.row_id.str.extract(r'^track_3/e(\d+)/w\d+/f\d+/o(\d+)$').astype(int); k.columns = ['ep', 'slot']
diff = (a.object_mesh_b64.fillna('') != b.object_mesh_b64.fillna('')).values
names = {}
for ep in sorted(k.ep.unique()):
    r = pd.read_parquet(f'{A}/results/episode_{ep:06d}.parquet'); names[int(ep)] = [n for _, n in r.drop_duplicates('object_id').sort_values('object_id')[['object_id', 'object_body_name']].values.tolist()]
out['mesh_rows_changed'] = int(diff.sum()); out['mesh_changed_ep_obj'] = sorted({f'{int(e)}:{names[int(e)][int(s)]}' for e, s in k[diff].values})
pose = [c for c in cols if c not in ('row_id',)]
rowdiff = (a[pose].astype(str) != b[pose].astype(str)).any(axis=1).values if out['row_ids_equal'] else None
out['episodes_with_non_mesh_row_changes'] = sorted({int(e) for e in k[rowdiff].ep.values}) if rowdiff is not None else None
open(OUT, 'w').write(json.dumps(out, indent=1)); print(json.dumps({x: out[x] for x in out if x != 'mesh_changed_ep_obj'}), '\nmesh changed:', out['mesh_changed_ep_obj'])
