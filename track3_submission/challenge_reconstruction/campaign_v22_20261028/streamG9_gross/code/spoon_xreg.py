"""Spoon cross-mesh registration consistency (kit _register_object_body, kit seeds of every spoon episode + random seeds).
All 8 'main' spoon episodes submit the same file, so the official registration differs only by the per-episode seed.
For each (candidate, reference) pair: rotation per seed, basin clustering, inter-basin angle and roll axis, symmetric distance."""
import sys, json, itertools
import numpy as np, pandas as pd, trimesh
sys.path.insert(0, '/data1/SRL_V2D/challenge_submissions_SRL/analysis_20261020/scripts')
import public_kit_score as K
M = K.load_metric('MP-SR')
P = '/data1/SRL_V2D/challenge_submissions_SRL/track3_v24_20261028/all_20_episodes'
sub = pd.read_parquet(f'{P}/track3_submission.parquet', columns=['row_id', 'object_mesh_b64'])
sub = sub[sub.object_mesh_b64.fillna('').str.strip() != '']
def pmesh(ep, slot):
    r = sub[sub.row_id.str.contains(f'/e{ep:06d}/') & sub.row_id.str.endswith(f'/o{slot}')]
    return M._decode_object_mesh(r.object_mesh_b64.iloc[0])
def load(p):
    m = trimesh.load(p, force='mesh'); m.merge_vertices(); m.remove_unreferenced_vertices(); return np.asarray(m.vertices, float), np.asarray(m.faces, np.int64)
slots = {3: 1, 5: 1, 6: 1, 15: 1, 16: 1, 32: 1, 33: 1, 35: 0, 36: 0, 37: 0}
res = pd.read_parquet  # noqa
import glob, os
for ep in slots:  # verify slot from results roster
    r = pd.read_parquet(f'{P}/results/episode_{ep:06d}.parquet'); names = [n for _, n in r.drop_duplicates('object_id').sort_values('object_id')[['object_id', 'object_body_name']].values.tolist()]
    slots[ep] = names.index('wooden_spoon')
meshes = {'ours_5fcd(ep3)': pmesh(3, slots[3]), 'ours_e6c9(ep15)': pmesh(15, slots[15]),
          'compose27': load('/data1/SRL_V2D/challenge_reconstruction/campaign_v23_20261028/target_compose27/episode_000003/wooden_spoon_mesh_metric.obj'),
          'compose21': load('/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/compose_target/episode_000003/wooden_spoon_mesh_metric.obj')}
seeds = {f'ep{e}': 1_000_003 * e + 101 * s for e, s in slots.items()}
rng = np.random.default_rng(0); seeds.update({f'r{i}': int(rng.integers(1, 2**31)) for i in range(14)})
def ang(A, B): return float(np.degrees(np.arccos(np.clip((np.trace(A.T @ B) - 1) / 2, -1, 1))))
out = {}
for cn, rn in itertools.permutations(meshes, 2):
    if cn.startswith('compose') and rn.startswith('compose'): continue
    cand, ref = meshes[cn], meshes[rn]; rc, _ = M._surface_moments(*ref)
    Rs = {k: M._register_object_body(cand, ref, rc, s)[1] for k, s in seeds.items()}
    base = Rs['ep3']; angs = {k: round(ang(base, R), 1) for k, R in Rs.items()}
    far = [k for k, a in angs.items() if a > 30]
    entry = dict(angle_to_ep3_seed=angs, n_other_basin=len(far), other_basin=far)
    if far:
        D = base.T @ Rs[far[0]]; rv = Rot = None
        from scipy.spatial.transform import Rotation as RR
        rv = RR.from_matrix(D).as_rotvec(); entry['basin_rotvec_deg'] = (np.degrees(rv)).round(1).tolist()
    # spoon long axis in candidate body frame (PCA major) and roll angle between basins
    out[f'{cn} -> {rn}'] = entry
    print(cn, '->', rn, 'other-basin seeds:', far, 'max angle', max(angs.values()), flush=True)
json.dump(out, open(sys.argv[1], 'w'), indent=1)
