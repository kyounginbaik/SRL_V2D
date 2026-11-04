"""E31 cwmesh: per (episode, object) decide how the coworker bundle's own mesh enters a component whose robot poses are in the component's
body frame. Kit registration (MP-SR._register_object_body: surface moments + 3000-sample rigid alignment), candidate = bundle mesh,
reference = the v29 component mesh (for G18 spoons the v28 component spoon, whose body frame G18 shares by construction), 10 seeds. Map (no rescale, coworker size kept): x_comp = R (x_bundle - c_bundle) + c_comp + offset.
Best seed = lowest surface fit residual (bundle samples -> component surface). If its angle <= 0.5 deg (symmetric objects: symmetry-axis angle) and the centroid moves <= 1 mm -> byte-exact bundle file; else the bundle geometry re-expressed
(vertices rigidly transformed, faces / vertex order unchanged). Twin check: number of seeds within 2 deg of the best (symmetry-aware).
Bundle meshes are loaded as data only (trimesh OBJ parse). usage: cwmesh_plan.py BUNDLE OUTDIR"""
import sys, json, hashlib, shutil
from pathlib import Path
import numpy as np, trimesh
sys.path.insert(0, '/data1/SRL_V2D/challenge_submissions_SRL/analysis_20261020/scripts')
import public_kit_score as K
M = K.load_metric('MP-SR')
from scipy.spatial.transform import Rotation as Rot
C = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028'); CA = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates')
X, OUT = Path(sys.argv[1]), Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=False)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    m = trimesh.load(p, force='mesh', process=False); return np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64), m
s29 = json.loads((C / 'selection_v29.json').read_text()); s28 = json.loads((C / 'selection_v28.json').read_text()); rec = {}
for e, tag in sorted(s29.items(), key=lambda kv: int(kv[0])):
    n = f'episode_{int(e):06d}'; mj = json.loads((X / n / 'meshes.json').read_text()); (OUT / n).mkdir()
    for name, v in mj.items():
        b = X / n / v['file']; c = CA / tag / n / 'meshes' / n / f'{name}.obj'
        if name == 'wooden_spoon' and tag.startswith('G18_'):   # the G18 spoon was built in the v28 spoon's body frame (identity): the frame-defining mesh is the v28 one
            c = CA / s28[e] / n / 'meshes' / n / f'{name}.obj'
        bv, bf, bm = load(b); cv, cf, _ = load(c)
        cc, _ = M._surface_moments(bv, M._mesh_faces(bf.astype(np.float64), len(bv))); rc, _ = M._surface_moments(cv, M._mesh_faces(cf.astype(np.float64), len(cv)))
        from scipy.spatial import cKDTree
        cmesh = trimesh.Trimesh(cv, cf, process=False); bmesh = trimesh.Trimesh(bv, bf, process=False)
        tree = cKDTree(cmesh.sample(20000, seed=7)); bpts = bmesh.sample(5000, seed=8)
        ax, order = K.symmetry_axis(name, cv); ax = np.asarray(ax, float); sym = bool(np.linalg.norm(ax) > 0 and int(order) != 1)
        fit = lambda R_, t_: float(tree.query(bpts @ R_.T + t_)[0].mean() * 1000)
        seeds = []
        for seed in range(10):
            ca, R_ = M._register_object_body((bv, M._mesh_faces(bf.astype(np.float64), len(bv))), (cv, M._mesh_faces(cf.astype(np.float64), len(cv))), rc, 1000 + seed)
            t_ = rc - R_ @ ca   # candidate anchor ca (bundle coords) <-> reference anchor rc (component coords): x_comp = R (x_b - ca) + rc
            seeds.append((fit(R_, t_), R_, t_))
        seeds.sort(key=lambda z: z[0]); res, R, t = seeds[0]
        def gap(Ra, Rb):   # twin distance; symmetric objects (kit symmetry axis in the component frame): angle between the mapped axes only
            if sym: return float(np.degrees(np.arccos(np.clip(np.dot(Ra.T @ ax, Rb.T @ ax), -1, 1))))   # signed: an upside-down twin counts as a different twin
            return float(np.degrees(Rot.from_matrix(Ra @ Rb.T).magnitude()))
        same = sum(gap(z[1], R) <= 2.0 for z in seeds); spread = max(gap(z[1], R) for z in seeds)
        ang = gap(R, np.eye(3)) if sym else float(np.degrees(Rot.from_matrix(R).magnitude())); tr = float(np.linalg.norm(t) * 1000)
        dc = float(np.linalg.norm(R @ cc + t - cc) * 1000); id_res = fit(np.eye(3), np.zeros(3))
        method = 'kit_registration_best_seed'; corr = None
        if bf.shape == cf.shape and np.array_equal(bf, cf):
            # derived mesh (identical topology): the frame relation is measured exactly by vertex correspondence (similarity Procrustes x_c ~ s R x_b + t);
            # the coworker geometry is kept (no scale applied); the kit registration (moment-centred) is reported but not used for these
            mb, mc = bv.mean(0), cv.mean(0); A0, B0 = bv - mb, cv - mc; U, S_, Vt = np.linalg.svd(A0.T @ B0); D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
            Rc = (U @ D @ Vt).T; sc = float(np.trace(np.diag(S_) @ D) / (A0 ** 2).sum()); tc = mc - sc * Rc @ mb
            corr = dict(rot_deg=float(np.degrees(Rot.from_matrix(Rc).magnitude())), scale=sc, origin_offset_mm=float(np.linalg.norm(tc) * 1000), rms_mm=float(np.sqrt(((sc * A0 @ Rc.T - B0) ** 2).sum(1).mean()) * 1000))
            method = 'vertex_correspondence'
            exact = corr['rot_deg'] <= 0.5 and corr['origin_offset_mm'] <= 1.0
            R, t = Rc, tc   # rigid part (scale not applied: coworker size kept)
            if sym: ang = float(np.degrees(np.arccos(np.clip(np.dot(Rc.T @ ax, ax), -1, 1))))
            else: ang = corr['rot_deg']
            dc = corr['origin_offset_mm']
        else:
            exact = ang <= 0.5 and dc <= 1.0
        dst = OUT / n / f'{name}.obj'
        if exact:
            shutil.copyfile(b, dst); mode = 'byte_exact'
        else:
            nv = bv @ R.T + t
            m2 = trimesh.Trimesh(vertices=nv, faces=bf, process=False); m2.export(dst); mode = 're_expressed'
        # identity-vs-transform check: mean nearest distance of the shipped mesh to the component mesh
        sv = trimesh.load(dst, force='mesh', process=False)
        cm = trimesh.Trimesh(cv, cf, process=False); pts = sv.sample(4000, seed=1) if hasattr(sv, 'sample') else sv.vertices
        d = float(np.abs(trimesh.proximity.signed_distance(cm, pts[:1500])).mean() * 1000) if False else float(np.linalg.norm(pts.mean(0) - cm.vertices.mean(0)) * 1000)
        r = dict(mode=mode, method=method, correspondence=corr, rot_deg=round(ang, 3), trans_mm=round(tr, 3), centroid_move_mm=round(dc, 3), seed_spread_deg=round(float(spread), 3), seeds_on_best_twin=int(same), twin_consistent=bool(same == 10), symmetric=sym, fit_mm=round(res, 3), identity_fit_mm=round(id_res, 3), seed_fits_mm=[round(z[0], 3) for z in seeds],
                 bundle=str(b), bundle_sha256=sha(b), component_mesh=str(c), component_sha256=sha(c), shipped_sha256=sha(dst), bundle_nv=len(bv), shipped_nv=len(sv.vertices),
                 R=R.round(8).tolist(), t=t.round(8).tolist(), centroid_gap_to_component_mm=round(d, 2))
        rec[f'{e}:{name}'] = r
        print(e, name, mode, method[:6], f"rot {ang:.2f} deg dc {dc:.2f} mm twin {same}/10 sym {sym} fit {res:.2f} mm (identity {id_res:.2f}) seedfits {seeds[0][0]:.2f}..{seeds[-1][0]:.2f} nv {len(bv)}->{len(sv.vertices)}", flush=True)
(OUT / 'cwmesh_plan.json').write_text(json.dumps(rec, indent=1))
