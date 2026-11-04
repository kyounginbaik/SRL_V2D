"""Per-frame robot-vs-recon errors for one Track 3 episode, with the kit's unchanged registration and scene alignment.
Same steps as score_vs_compose.py + public_kit_score.aligned_errors, but keeps the per-frame arrays instead of the means.
usage: ep33_perframe.py OUT_JSON EP LABEL:SUBMISSION_PARQUET:TARGET_DIR [...]"""
import sys, json
import numpy as np, pandas as pd
import trimesh
sys.path.insert(0, '/data1/SRL_V2D/challenge_submissions_SRL/analysis_20261020/scripts')
import public_kit_score as K
OUT, EP, RUNS = sys.argv[1], int(sys.argv[2]), [a.split(':') for a in sys.argv[3:]]
M = K.load_metric('MP-SR')

def ref_geometry(names, paths):  # copied from score_vs_compose.py (that module runs its scoring loop at import)
    g = dict(names=np.array(names))
    for i, (n, p) in enumerate(zip(names, paths)):
        m = trimesh.load(p, force='mesh'); m.merge_vertices()
        if len(m.faces) > 40000:
            try: m = m.simplify_quadric_decimation(face_count=40000)
            except Exception: pass
        m.remove_unreferenced_vertices(); v = np.asarray(m.vertices, np.float64); f = np.asarray(m.faces, np.int64)
        a, _ = M._surface_moments(v, f); ax, o = K.symmetry_axis(n, v)
        g.update({f'v{i}': v, f'f{i}': f, f'anchor{i}': a, f'points{i}': M._sample_mesh_surface(v, f, 10000, 0), f'axis{i}': ax, f'order{i}': np.int64(o)})
    return g
WINDOWS = [(0, 40), (40, 80), (80, 124), (124, 137), (137, 172), (172, 181), (181, 220), (220, 260), (260, 10**6)]

def perframe(M, sol, sub):
    geometry = M._decode_archive(sol['reference_geometry_b64'].iloc[0], 16_000_000); lookup = {str(n): i for i, n in enumerate(geometry['names'])}
    merged = sol.merge(sub[['row_id', *M._REQUIRED_SUBMISSION_COLUMNS, 'object_mesh_b64']], on='row_id')
    achieved, reference, _, names = M._episode_tensors(merged)
    achieved = M._normalise_quaternions(achieved, 'submission', ValueError).copy(); reference = M._normalise_quaternions(reference, 'solution', ValueError).copy()
    axes, orders = [], []
    for slot, name in enumerate(names):
        rows = merged[merged.object_slot == slot].sort_values(['world_index', 'frame_index']); enc = rows.object_mesh_b64.fillna('').astype(str)
        cand = M._decode_object_mesh(enc[enc.str.strip() != ''].iloc[0]); i = lookup[str(name)]
        mesh = (geometry[f'v{i}'], geometry[f'f{i}'].astype(np.int64)); anchor = geometry[f'anchor{i}']
        ca, rot = M._register_object_body(cand, mesh, anchor, 1_000_003 * int(merged.episode_index.iloc[0]) + 101 * slot)
        achieved[:, :, slot] = M._convert_object_body(achieved[:, :, slot], ca, rot)
        reference[:, :, slot, :3] += np.einsum('...ij,j->...i', M._rotation_matrix(reference[:, :, slot, 3:]), anchor)
        axes.append(geometry[f'axis{i}']); orders.append(int(geometry[f'order{i}']))
    achieved = M._align_registered_scene(achieved, reference, axes)
    pos = np.linalg.norm(achieved[..., :3] - reference[..., :3], axis=-1)[:, 0] * 100
    ori = np.degrees(M._symmetry_orientation_errors(achieved, reference, axes, orders))[:, 0]
    return [str(n) for n in names], pos, ori

out = {}
for label, sub_path, tgt in RUNS:
    sub_all = pd.read_parquet(sub_path); sub_all['ep'] = sub_all.row_id.str.extract(r'/e(\d+)/')[0].astype(int)
    sub = sub_all[sub_all.ep == EP].drop(columns='ep').copy()
    res = sub_path.rsplit('/', 1)[0] + f'/results/episode_{EP:06d}.parquet'
    r = pd.read_parquet(res); names = [n for _, n in r.drop_duplicates('object_id').sort_values('object_id')[['object_id', 'object_body_name']].values.tolist()]
    d = f'{tgt}/episode_{EP:06d}'
    tracks = [np.load(f'{d}/{n}_trajectory.npz')['T_world_obj'] for n in names]; meshes = [f'{d}/{n}_mesh_metric.obj' for n in names]
    rows = []
    for _, s in sub.iterrows():
        f = int(s.row_id.split('/f')[1][:6]); slot = int(s.row_id.split('/o')[1])
        T = tracks[slot][min(f, len(tracks[slot]) - 1)]; c = K.pose_cols(T[None])
        rows.append(dict(row_id=s.row_id, episode_index=EP, world_index=0, frame_index=f, object_slot=slot, object_name=names[slot],
                         **{'ref_' + k: float(v[0]) for k, v in c.items()}, reference_geometry_b64=''))
    sol = pd.DataFrame(rows); sol.loc[0, 'reference_geometry_b64'] = K.encode(**ref_geometry(names, meshes))
    nm, pos, ori = perframe(M, sol, sub); n = len(pos)
    res_l = {}
    for k, name in enumerate(nm):
        p, o = pos[:, k], ori[:, k]
        res_l[name] = dict(mean_cm=round(float(p.mean()), 2), mean_deg=round(float(o.mean()), 1), max_deg=round(float(o.max()), 1),
                           frac_deg_gt15=round(float((o > 15).mean()), 3), frac_deg_gt30=round(float((o > 30).mean()), 3),
                           windows={f'f{a}-{min(b, n) - 1}': [round(float(p[a:b].mean()), 2), round(float(o[a:b].mean()), 1)] for a, b in WINDOWS if a < n},
                           deg=[round(float(x), 1) for x in o], cm=[round(float(x), 2) for x in p])
    out[label] = dict(submission=sub_path, target=tgt, frames=n, objects=res_l)
    print(label, json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ('deg', 'cm')} for k, v in res_l.items()}), flush=True)
json.dump(out, open(OUT, 'w'), indent=1)
