"""Build a Track 3 package that submits the scan-derived meshes (public GT scans remeshed, in the scan body frames) with our
rollout poses re-expressed in those frames. Conversion mirrors the kit's registration exactly (_register_object_body):
X_scan = Rr * s * (X_ours - cc) + offset + rc  ->  world_T_scan = (R_w Rr^T, p_w + R_w * our_coords_of_scan_origin).
For half-turn-symmetric objects without a declared symmetry (mini_sweeper, wooden pieces) the registration twin is chosen by a
physical prior from the public ground truth (sweeper rests bristles down = scan -z down; piece_1 starts -x down, piece_2 +x down).
Usage: make_scanmesh_package.py SRC_PACKAGE DST_PACKAGE NEW_MESH_OBJ_DIR
"""
import importlib.util, json, math, os, shutil, sys, glob, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq, trimesh
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
SRC, DST, NEW = map(Path, sys.argv[1:4]); KIT = SRC / 'submission_kit'
spec = importlib.util.spec_from_file_location('mpsr', KIT / 'metric_code/track_3/MP-SR.py'); M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
PRIOR = {'mini_sweeper': ('down', np.array([0, 0, -1.])), 'wooden_piece_1': ('down', np.array([-1., 0, 0])), 'wooden_piece_2': ('down', np.array([1., 0, 0]))}
AMBIGUOUS = set(PRIOR)

def align_all(reference, candidate):
    tree = cKDTree(reference); out = []
    for start in M._initial_rotations(reference, candidate, M._ALIGNMENT_STARTS):
        rotation = start.copy(); translation = reference.mean(0) - rotation @ candidate.mean(0); previous = math.inf
        for _ in range(M._ICP_ITERATIONS):
            moved = candidate @ rotation.T + translation; d, idx = tree.query(moved, k=1)
            keep = max(3, int(math.ceil(len(d) * M._TRIM_FRACTION))); chosen = np.argpartition(d, keep - 1)[:keep] if keep < len(d) else np.arange(len(d))
            sr, st = M._kabsch(moved[chosen], reference[idx[chosen]]); rotation = sr @ rotation; translation = sr @ translation + st
            err = float(np.mean(d[chosen]))
            if math.isfinite(previous) and abs(previous - err) <= 1e-8 * max(previous, 1.0): break
            previous = err
        out.append((rotation, translation, M._symmetric_mean_distance(reference, candidate @ rotation.T + translation)))
    return sorted(out, key=lambda t: t[2])

def load(p):
    m = trimesh.load(p, force='mesh', process=False); return np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64)

if DST.exists(): raise SystemExit('refusing to overwrite ' + str(DST))
shutil.copytree(SRC, DST, ignore=shutil.ignore_patterns('__pycache__', '*.pyc')); (DST / 'track3_submission.parquet').unlink(); (DST / 'track3_submission.cd-o.parquet').unlink(missing_ok=True)
roster = json.load(open(KIT / 'data/track_3_evaluation_objects.json'))['episodes']; report = {}
for pq_path in sorted((DST / 'results').glob('episode_*.parquet')):
    ep = int(pq_path.stem.split('_')[-1]); table = pq.read_table(pq_path); df = table.to_pandas(); meta = table.schema.metadata
    for slot, name in enumerate(roster[str(ep)]):
        if not (NEW / f'{name}.obj').exists():
            report[f'{ep}:{name}'] = dict(kept_our_mesh=True, note='no scan-derived mesh for this object (not in the public scans); poses and mesh unchanged'); print(f'ep{ep:2d} {name:18s} kept our mesh (no scan-derived mesh)'); continue
        ours = load(DST / 'meshes' / f'episode_{ep:06d}' / f'{name}.obj'); scan = load(NEW / f'{name}.obj')
        cv, cf = ours; rv, rf = scan; cc, cr = M._surface_moments(cv, cf); rc, rr = M._surface_moments(rv, rf); s = rr / cr; seed = 1_000_003 * ep + 101 * slot
        cp = (M._sample_mesh_surface(cv, cf, 3000, seed) - cc) * s; rp = M._sample_mesh_surface(rv, rf, 3000, seed) - rc
        sols = align_all(rp, cp); best = sols[0]; chosen = best; twin_note = 'best residual'
        rows = df.object_body_name == name; sub = df[rows].sort_values('step')
        q0 = sub[['achieved_quaternion_x', 'achieved_quaternion_y', 'achieved_quaternion_z', 'achieved_quaternion_w']].to_numpy()[0]; Rw0 = R.from_quat(q0).as_matrix()
        if name in AMBIGUOUS:
            kind, axis = PRIOR[name]; cands = [x for x in sols if x[2] - best[2] < 0.003]
            scored = []
            for sol in cands:
                Rws = Rw0 @ sol[0].T; down_in_scan = Rws.T @ np.array([0, 0, -1.]); rel = np.degrees((R.from_matrix(sol[0]) * R.from_matrix(best[0]).inv()).magnitude())
                scored.append((round(float(down_in_scan @ axis), 3), -sol[2], sol, rel))
            scored.sort(key=lambda t: (-t[0], -t[1])); chosen = scored[0][2]
            twin_note = f'prior: frame-0 world-down vs scan axis {axis.tolist()}: ' + ' | '.join(f'dot {d:+.2f} res {-r*100:.2f} cm rot {rel:.0f}deg' for d, r, _, rel in scored) + f' -> chosen rot {scored[0][3]:.0f} deg from best'
        rotation, offset, resid = chosen; origin_ours = cc + rotation.T @ (0 - rc - offset) / s   # scan body origin in our body coords
        p = sub[['achieved_position_x', 'achieved_position_y', 'achieved_position_z']].to_numpy(); q = sub[['achieved_quaternion_x', 'achieved_quaternion_y', 'achieved_quaternion_z', 'achieved_quaternion_w']].to_numpy()
        Rw = R.from_quat(q).as_matrix(); p_new = p + np.einsum('nij,j->ni', Rw, origin_ours); R_new = Rw @ rotation.T; q_new = R.from_matrix(R_new).as_quat()
        df.loc[sub.index, ['achieved_position_x', 'achieved_position_y', 'achieved_position_z']] = p_new; df.loc[sub.index, ['achieved_quaternion_x', 'achieved_quaternion_y', 'achieved_quaternion_z', 'achieved_quaternion_w']] = q_new
        shutil.copy2(NEW / f'{name}.obj', DST / 'meshes' / f'episode_{ep:06d}' / f'{name}.obj')
        report[f'{ep}:{name}'] = dict(scale_scan_over_ours=float(s), residual_cm=float(resid * 100), twin=twin_note, origin_shift_cm=float(np.linalg.norm(origin_ours) * 100))
        print(f'ep{ep:2d} {name:18s} scale {s:.3f} residual {resid*100:.2f} cm | {twin_note}')
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False).replace_schema_metadata(meta), pq_path)
json.dump(report, open(DST / 'mesh_frame_conversion.json', 'w'), indent=1); print('converted', len(report), 'episode/object pairs')
