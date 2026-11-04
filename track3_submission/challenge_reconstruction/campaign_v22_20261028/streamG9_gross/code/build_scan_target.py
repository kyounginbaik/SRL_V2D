"""SCORER-ONLY validation target: public scan (or, for the spoon, a surrogate mesh) mapped into OUR final_v6 body frame by the
kit's own _register_object_body, trajectories = final_v6 poses. Scoring a package against it simulates the official mesh
registration if our reconstruction were exact. The scan never leaves this scorer folder.
usage: build_scan_target.py V6_TARGET OUT SPOON_SURROGATE_DIR(ep dirs with wooden_spoon_mesh_metric.obj) [SEED]"""
import sys, json, glob, os, shutil
import numpy as np, trimesh
sys.path.insert(0, '/data1/SRL_V2D/challenge_submissions_SRL/analysis_20261020/scripts')
import public_kit_score as K
M = K.load_metric('MP-SR')
V6, OUT, SP = sys.argv[1:4]; SEED = 0; ONLY = sys.argv[4] if len(sys.argv) > 4 else None
def load(p, cap=40000):
    m = trimesh.load(p, force='mesh'); m.merge_vertices()
    if len(m.faces) > cap:
        try: m = m.simplify_quadric_decimation(face_count=cap)
        except Exception: pass
    m.remove_unreferenced_vertices(); return np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64)
rep = {}
for ed in sorted(glob.glob(f"{V6}/episode_*" if ONLY is None else f"{V6}/episode_{int(ONLY):06d}")):
    ep = os.path.basename(ed); od = f'{OUT}/{ep}'; os.makedirs(od, exist_ok=True)
    for tp in sorted(glob.glob(f'{ed}/*_trajectory.npz')):
        name = os.path.basename(tp)[:-len('_trajectory.npz')]
        shutil.copy2(tp, f'{od}/{name}_trajectory.npz')
        ours = load(f'{ed}/{name}_mesh_metric.obj')
        scan_p = f'{SP}/{ep}/wooden_spoon_mesh_metric.obj' if name == 'wooden_spoon' else K.gt_mesh_path(name)
        sv, sf = load(scan_p)
        rc, rr = M._surface_moments(sv, sf); cc, cr = M._surface_moments(*ours); scale = rr / cr
        res = []
        for s in range(SEED, SEED + 3):
            ca, R = M._register_object_body(ours, (sv, sf), rc, s); res.append((ca, R))
        ca, R = res[0]
        spread = [float(np.degrees(np.arccos(np.clip((np.trace(R.T @ r) - 1) / 2, -1, 1)))) for _, r in res]
        xv = ca + ((sv - rc) @ R) / scale   # x_v = ca + R^T (x_s - rc)/scale
        trimesh.Trimesh(xv, sf, process=False).export(f'{od}/{name}_mesh_metric.obj')
        rep[f'{ep}/{name}'] = dict(scan=scan_p, scale=float(scale), seed_rot_spread_deg=spread)
        print(ep, name, round(scale, 4), [round(x, 1) for x in spread], flush=True)
json.dump(rep, open(f"{OUT}/build_report_{ONLY}.json", "w"), indent=1)
