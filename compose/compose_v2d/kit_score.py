"""Score tracked trajectories with the NVIDIA kit's own Track 3 scorers (metric_code/track_3/{AUC,MPPE,MP-SR,SP-SR,RPE}.py).

Builds the `solution` frame the host holds (public GT mocap poses + reference geometry payload built from the public GT
scans: v/f, anchor = surface-moment centroid, 10k surface samples, symmetry axis/order) and the `submission` frame the
packer would produce (our poses, provenance columns, object_mesh_b64 = v2dlb.mesh_budget.budget_mesh of our metric mesh),
then calls `score(solution, submission, 'row_id')` of every metric module unchanged.
Caveat: the host's anchor / sample / symmetry-axis values are confidential; the code path is identical, those inputs are
our best reconstruction of them.

usage: kit_score.py --runs 41:white_pot:_meshicp:compose_meshicp [45:white_pot:_meshicp:compose_meshicp ...]
       (several runs of the same episode = several objects of that episode; different episodes are averaged like Kaggle)
"""
import argparse, base64, importlib.util, io, os, sys
import numpy as np, pandas as pd, trimesh
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, T3, work_dir, gt_world_T_obj, gt_mesh_path
KIT = ROOT / 'third_party/v2d_submission_kit'
sys.path.insert(0, str(KIT))
from v2dlb.mesh_budget import budget_mesh

# symmetry (host-side in the real scorer): axis in the GT scan frame, order 360 = any axial rotation, 2 = half turn
SYMMETRY = {'beige_cup': ('pca_min', 360), 'blue_cup': ('pca_min', 360), 'white_pot_lid': ('pca_min', 2)}
PROV = dict(eval_script_sha256='0' * 64, eval_code_sha256='0' * 64, checkpoint_sha256='0' * 64, packer_script_sha256='0' * 64,
            uploader_script_sha256='0' * 64, code_commit_url='https://github.com/visualailab/compose-v2d/commit/0123456789abcdef0123456789abcdef01234567')


def load_metric(name):
    spec = importlib.util.spec_from_file_location('v2d_' + name.replace('-', '_'), KIT / 'metric_code/track_3' / f'{name}.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def encode(**arrays):
    s = io.BytesIO(); np.savez_compressed(s, **arrays); return base64.b64encode(s.getvalue()).decode('ascii')


def symmetry_axis(name, v):
    """Axis of revolution = the principal axis whose eigenvalue is farthest from the other two (a body of revolution has two
    equal radial eigenvalues). Host values are confidential; this is our reconstruction of them."""
    if name not in SYMMETRY: return np.zeros(3), 1
    _, order = SYMMETRY[name]
    c = v - v.mean(0); w, U = np.linalg.eigh(c.T @ c / len(c))
    gaps = [abs(w[i] - w[(i + 1) % 3]) + abs(w[i] - w[(i + 2) % 3]) - abs(w[(i + 1) % 3] - w[(i + 2) % 3]) for i in range(3)]
    axis = U[:, int(np.argmax(gaps))]
    return axis / np.linalg.norm(axis), order


def reference_geometry(names, auc):
    g = dict(names=np.array(names))
    for i, n in enumerate(names):
        m = trimesh.load(gt_mesh_path(n), force='mesh')
        m.merge_vertices()                                        # glb scans carry duplicated seam vertices (V > F)
        cap = min(40000, int((14_000_000 / len(names) - 300_000) / 60))   # host payload limit is 16 MB decoded for ALL objects (~60 B/face + 10k samples)
        if len(m.faces) > cap:
            try: m = m.simplify_quadric_decimation(face_count=cap)
            except Exception: pass
        m.remove_unreferenced_vertices()
        v = np.asarray(m.vertices, np.float64); f = np.asarray(m.faces, np.int64)
        anchor, _ = auc._surface_moments(v, f)
        axis, order = symmetry_axis(n, v)
        g[f'v{i}'] = v; g[f'f{i}'] = f; g[f'anchor{i}'] = anchor; g[f'points{i}'] = auc._sample_mesh_surface(v, f, 10000, 0)
        g[f'axis{i}'] = axis; g[f'order{i}'] = np.int64(order)
    return g


def pose_cols(T):
    q = Rot.from_matrix(T[:, :3, :3]).as_quat()      # xyzw
    return dict(pos_x=T[:, 0, 3], pos_y=T[:, 1, 3], pos_z=T[:, 2, 3], quat_x=q[:, 0], quat_y=q[:, 1], quat_z=q[:, 2], quat_w=q[:, 3])


def build_episode(ep, runs, auc, mesh_src='ours'):
    """runs: list of (obj, tag, variant). Returns (solution_df, submission_df) for this episode (geometry payload added later)."""
    sol_rows = []; sub_rows = []
    for slot, (obj, tag, variant) in enumerate(runs):
        gt, _ = gt_world_T_obj(ep, obj)
        z = np.load(work_dir(ep, obj) / f'track{tag}' / 'poses.npz'); ours = z[variant]
        n = min(len(gt), len(ours))
        gcols = pose_cols(gt[:n]); ocols = pose_cols(ours[:n])
        mesh_path = (work_dir(ep, obj) / f'track{tag}' / 'mesh_metric.obj') if mesh_src == 'ours' else gt_mesh_path(obj)
        mv, mf = budget_mesh(mesh_path, 4096, 4096)
        payload = encode(vertices=np.asarray(mv, np.float64), faces=np.asarray(mf, np.float64))
        for f in range(n):
            rid = f'track_3/e{ep:06d}/w000/f{f:06d}/o{slot}'
            sol_rows.append(dict(row_id=rid, episode_index=ep, world_index=0, frame_index=f, object_slot=slot, object_name=obj,
                                 **{'ref_' + k: float(v[f]) for k, v in gcols.items()}, reference_geometry_b64=''))
            sub_rows.append(dict(row_id=rid, **{k: float(v[f]) for k, v in ocols.items()}, **PROV,
                                 object_mesh_b64=payload if f == 0 else ''))
    return pd.DataFrame(sol_rows), pd.DataFrame(sub_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True, help="ep:obj:tag:variant ... or 'public_all' (with --tag --variant)")
    ap.add_argument('--tag', default='_v3'); ap.add_argument('--variant', default='compose_meshicp_smooth')
    ap.add_argument('--mesh', default='ours', choices=['ours', 'gt'], help="submitted mesh: our metric mesh or the GT scan (oracle)")
    ap.add_argument('--per_episode', action='store_true', help='also print each episode separately')
    ap.add_argument('--tags', nargs='*', default=None, help="public_best: restrict the per-object choice to these tags (default: every tag present)")
    ap.add_argument('--variants', nargs='*', default=['icp_track_6dof', 'compose_meshicp_smooth', 'compose_meshicp'], help='public_best: candidate variants')
    a = ap.parse_args()
    if a.runs == ['public_best']:                                  # per object: tag/variant with the best in-script ADD-AUC (GT needed -> public only)
        import json
        from v2d_io import PUBLIC_EPISODES, episode_objects
        a.runs = []
        for ep in PUBLIC_EPISODES:
            picks = []
            for o in episode_objects(ep):
                best = None
                for td in sorted(work_dir(ep, o).glob('track_*')):
                    tag = td.name[5:]
                    if (a.tags and tag not in a.tags) or not (td / 'poses.npz').exists() or not (td / 'results.json').exists(): continue
                    res = json.load(open(td / 'results.json')).get('results', {})
                    for v in a.variants:
                        auc_ = res.get(v, {}).get('d0', {}).get('auc')
                        if auc_ is not None and (best is None or auc_ > best[0]): best = (auc_, tag, v)
                if best: picks.append(f'{ep}:{o}:{best[1]}:{best[2]}'); print(f'episode {ep} {o}: {best[1]}/{best[2]} (in-script AUC {best[0]:.3f})')
            if len(picks) == len(episode_objects(ep)): a.runs += picks
            else: print(f'episode {ep}: incomplete, skipped')
    if a.runs == ['public_all']:                                   # every public episode/object that has a tracking result
        from v2d_io import PUBLIC_EPISODES, episode_objects
        tag, variant = sys.argv[sys.argv.index('--tag') + 1], sys.argv[sys.argv.index('--variant') + 1]
        a.runs = []
        for ep in PUBLIC_EPISODES:
            objs = episode_objects(ep)
            if all((work_dir(ep, o) / f'track{tag}' / 'poses.npz').exists() for o in objs):
                a.runs += [f'{ep}:{o}:{tag}:{variant}' for o in objs]
            else:
                print(f'episode {ep}: incomplete ({[o for o in objs if not (work_dir(ep, o) / f"track{tag}" / "poses.npz").exists()]} missing), skipped')
    auc = load_metric('AUC')
    by_ep = {}
    for r in a.runs:
        ep, obj, tag, variant = r.split(':'); by_ep.setdefault(int(ep), []).append((obj, tag, variant))
    sols, subs = [], []
    for ep, runs in sorted(by_ep.items()):
        s, u = build_episode(ep, runs, auc, a.mesh); sols.append(s); subs.append(u)
        print(f'episode {ep}: {[r[0] for r in runs]} ({len(s)} rows)')
    solution = pd.concat(sols, ignore_index=True); submission = pd.concat(subs, ignore_index=True)
    names = sorted({r[0] for runs in by_ep.values() for r in runs})                 # one host geometry payload for all objects
    solution.loc[0, 'reference_geometry_b64'] = encode(**reference_geometry(names, auc))
    print(f'\nkit scores ({a.mesh} mesh), averaged over {len(by_ep)} episode(s):')
    mods = {name: load_metric(name) for name in ['AUC', 'SP-SR', 'MP-SR', 'RPE', 'MPPE']}
    def score_frames(sol, sub):
        out = {}
        for name, m in mods.items():
            try: out[name] = m.score(sol.copy(), sub.copy(), 'row_id')
            except Exception as e: out[name] = f'ERR {type(e).__name__}: {str(e)[:80]}'
        return out
    r = score_frames(solution, submission)
    for name, val in r.items(): print(f'  {name:6s} {val:.4f}' if isinstance(val, float) else f'  {name:6s} {val}')
    if a.per_episode and len(by_ep) > 1:
        print('\nper episode:')
        for s_, u_ in zip(sols, subs):
            ep = int(s_['episode_index'].iloc[0]); s_ = s_.copy(); s_.loc[s_.index[0], 'reference_geometry_b64'] = solution.loc[0, 'reference_geometry_b64']
            r = score_frames(s_, u_); objs = ','.join(sorted(set(s_['object_name'])))
            print(f'  ep{ep:3d} {objs:34s} ' + '  '.join(f'{k} {v:.3f}' if isinstance(v, float) else f'{k} ERR' for k, v in r.items()))


if __name__ == '__main__':
    main()
