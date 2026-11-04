"""Score a Track 3 submission parquet with the kit's unchanged scorers, using the coworker's compose_eval tracks + meshes as a
stand-in ground truth (registration of our mesh to theirs, anchor frame-0 alignment, exactly as the official scorer does).
usage: score_vs_compose.py SUBMISSION_PARQUET RESULTS_DIR COMPOSE_DIR OUT_JSON [EP ...]"""
import sys, json, glob, os
import numpy as np, pandas as pd, trimesh
sys.path.insert(0, '/data1/SRL_V2D/challenge_submissions_SRL/analysis_20261020/scripts')
import public_kit_score as K
SUB, RES, CMP, OUT = sys.argv[1:5]; EPS = [int(e) for e in sys.argv[5:]]
sub_all = pd.read_parquet(SUB); sub_all['ep'] = sub_all.row_id.str.extract(r'/e(\d+)/')[0].astype(int)
mods = {n: K.load_metric(n) for n in ['MP-SR', 'SP-SR', 'MPPE', 'AUC', 'RPE']}; M = mods['MP-SR']
def ref_geometry(names, paths):
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
out = {}
for res in sorted(glob.glob(f'{RES}/episode_*.parquet')):
    ep = int(res[-14:-8])
    if EPS and ep not in EPS: continue
    r = pd.read_parquet(res); order = r.drop_duplicates('object_id').sort_values('object_id')[['object_id', 'object_body_name']].values.tolist()
    names = [n for _, n in order]; d = f'{CMP}/episode_{ep:06d}'
    try:
        tracks = [np.load(f'{d}/{n}_trajectory.npz')['T_world_obj'] for n in names]; meshes = [f'{d}/{n}_mesh_metric.obj' for n in names]
    except FileNotFoundError as e:
        out[ep] = dict(error=str(e)); print(ep, 'missing', e); continue
    sub = sub_all[sub_all.ep == ep].drop(columns='ep').copy()
    rows = []
    for _, s in sub.iterrows():
        f = int(s.row_id.split('/f')[1][:6]); slot = int(s.row_id.split('/o')[1])
        T = tracks[slot][min(f, len(tracks[slot]) - 1)]; c = K.pose_cols(T[None])
        rows.append(dict(row_id=s.row_id, episode_index=ep, world_index=0, frame_index=f, object_slot=slot, object_name=names[slot],
                         **{'ref_' + k: float(v[0]) for k, v in c.items()}, reference_geometry_b64=''))
    sol = pd.DataFrame(rows); sol.loc[0, 'reference_geometry_b64'] = K.encode(**ref_geometry(names, meshes))
    sc = {}
    for k, m in mods.items():
        try: sc[k] = float(m.score(sol.copy(), sub.copy(), 'row_id'))
        except Exception as e: sc[k] = None; print(ep, k, 'ERR', str(e)[:150])
    try: err = K.aligned_errors(M, sol, sub)
    except Exception as e: err = {}; print(ep, 'align ERR', str(e)[:150])
    out[ep] = dict(scores=sc, objects={k: [round(a, 2), round(b, 1)] for k, (a, b) in err.items()}, track_frames=[len(t) for t in tracks], sub_frames=int(sol.frame_index.max()) + 1)
    print(ep, json.dumps(out[ep]), flush=True)
json.dump(out, open(OUT, 'w'), indent=1)
