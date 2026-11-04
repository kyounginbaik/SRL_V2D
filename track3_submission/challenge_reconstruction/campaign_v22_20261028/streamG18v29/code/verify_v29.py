"""v29 vs v28 parquet diff + packed spoon mesh check. usage: verify_v29.py V28_PARQUET V29_PARQUET V29_PKG_DIR"""
import sys, json, hashlib, numpy as np, pandas as pd
KIT = '/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261017/submission_readiness/latest_official/v2d_submission_kit'
sys.path.insert(0, KIT)
from v2dlb.track3_pose import validate_packed_meshes
from v2dlb.mesh_budget import budget_mesh
import trimesh
a, b = pd.read_parquet(sys.argv[1]), pd.read_parquet(sys.argv[2]); pkg = sys.argv[3]
out = dict(rows=(len(a), len(b)), row_ids_equal=bool((a.row_id.values == b.row_id.values).all()))
cols = [c for c in a.columns if c != 'object_mesh_b64']
out['non_mesh_columns_identical'] = {c: bool(a[c].equals(b[c])) for c in cols}
k = a.row_id.str.extract(r'^track_3/e(\d+)/w\d+/f\d+/o(\d+)$').astype(int); k.columns = ['ep', 'slot']
diff = a.object_mesh_b64.fillna('') != b.object_mesh_b64.fillna('')
roster = json.load(open(f'{pkg}/manifest.json'))
names = {}
for ep in sorted(k.ep.unique()):
    r = pd.read_parquet(f'{pkg}/results/episode_{ep:06d}.parquet'); names[int(ep)] = [n for _, n in r.drop_duplicates('object_id').sort_values('object_id')[['object_id', 'object_body_name']].values.tolist()]
changed = sorted({(int(e), names[int(e)][int(s)]) for e, s in k[diff].values})
out['mesh_rows_changed'] = int(diff.sum()); out['mesh_changed_ep_obj'] = changed
MA, MB = validate_packed_meshes(a), validate_packed_meshes(b)
G = '/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/streamG18_spoonmesh/mesh/'
spoon = {}
for (ep, slot), (v, f) in MB.items():
    if names[ep][slot] != 'wooden_spoon': continue
    src = G + ('wooden_spoon_g18_sub15_16.obj' if ep in (15, 16) else 'wooden_spoon_g18.obj')
    bv, bf = budget_mesh(src, 4096, 4096); raw = trimesh.load(src, force='mesh', process=False)
    ov, of = MA[(ep, slot)]
    pm = f'{pkg}/meshes/episode_{ep:06d}/wooden_spoon.obj'
    spoon[ep] = dict(packed_v=len(v), packed_f=len(f), g18_file_v=len(raw.vertices), g18_file_f=len(raw.faces),
                     equals_budget_of_g18=bool(np.allclose(v, bv, atol=1e-6) and np.array_equal(f, bf)),
                     max_abs_vs_g18_file_vertices=float(np.abs(np.asarray(v) - raw.vertices).max()) if len(v) == len(raw.vertices) else None,
                     faces_equal_g18_file=bool(len(f) == len(raw.faces) and np.array_equal(f, raw.faces)),
                     pkg_mesh_md5=hashlib.md5(open(pm, 'rb').read()).hexdigest(), g18_md5=hashlib.md5(open(src, 'rb').read()).hexdigest(),
                     v28_packed_v=len(ov), v28_packed_f=len(of))
out['spoon'] = spoon
print(json.dumps(out, indent=1))
