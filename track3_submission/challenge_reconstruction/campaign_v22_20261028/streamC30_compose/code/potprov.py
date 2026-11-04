"""C30 provenance diagnostic (scorer-side only; the public scan is read here to CLASSIFY a mesh, nothing is fitted or copied):
is the mesh used by the compose_eval_20261021 tracker for <ep>:<object> a copy of the public scan?
For each mesh: vertex/face count, extents, scale-normalised chamfer to the public scan after the kit registration.
Controls: compose ep20 water_pitcher (documented public-scan track 10-21/10-26), our own SRL meshes (team, GT-free).
usage: potprov.py OUT.json"""
import sys, json, numpy as np, trimesh
from pathlib import Path
from scipy.spatial import cKDTree
sys.path.insert(0, '/data1/SRL_V2D/challenge_submissions_SRL/analysis_20261020/scripts')
import public_kit_score as K
M = K.load_metric('MP-SR')
R = Path('/data1/SRL_V2D/challenge_reconstruction')
C21 = R / 'campaign_v16_20261023/compose_target'
C27 = R / 'campaign_v23_20261028/raw/compose_eval_20261027'
OURS = R / 'reconstruction_recovery_20261026/selected20_final_v14_g17'
FEM = Path('/data1/SRL_V2D/challenge_submissions_SRL/cowork_recon_gdrive_20261101/Jisu/final_episode_meshes_20261029')
items = []
for ep in (3, 8, 28, 5, 6, 1):
    items += [(f'c21_ep{ep}', 'white_pot', C21 / f'episode_{ep:06d}/white_pot_mesh_metric.obj'),
              (f'ours_ep{ep}', 'white_pot', OURS / f'episode_{ep:06d}/meshes/white_pot.obj')]
for ep in (3, 8, 28):
    items.append((f'fem1029_ep{ep}', 'white_pot', FEM / f'episode_{ep:06d}/white_pot_mesh_metric.obj'))
items += [('c21_ep20 (control: documented scan)', 'water_pitcher', C21 / 'episode_000020/water_pitcher_mesh_metric.obj'),
          ('c27_ep20', 'water_pitcher', C27 / 'episode_000020/water_pitcher_mesh_metric.obj'),
          ('ours_ep20', 'water_pitcher', OURS / 'episode_000020/meshes/water_pitcher.obj')]
ref = {}
def refg(n):
    if n not in ref:
        m = trimesh.load(K.gt_mesh_path(n), force='mesh'); m.merge_vertices(); m.remove_unreferenced_vertices()
        ref[n] = (np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64))
    return ref[n]
out = []
for lab, n, p in items:
    if not p.exists(): out.append(dict(label=lab, missing=str(p))); continue
    m = trimesh.load(p, force='mesh'); m.merge_vertices(); v = np.asarray(m.vertices, np.float64); f = np.asarray(m.faces, np.int64)
    rv, rf = refg(n)
    cc, cr = M._surface_moments(v, f); rc, rr = M._surface_moments(rv, rf)
    cp = M._sample_mesh_surface(v, f, 6000, 1) - cc; rp = M._sample_mesh_surface(rv, rf, 6000, 1) - rc
    best = None
    def kab(P, Q):  # R,t minimising |R Q + t - P|
        mp, mq = P.mean(0), Q.mean(0); U, S, Vt = np.linalg.svd((Q - mq).T @ (P - mp)); D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))
        Rm = Vt.T @ D @ U.T; return Rm, mp - Rm @ mq
    def pca(X): w, U = np.linalg.eigh(X.T @ X / len(X)); return U[:, ::-1]
    Ur = pca(rp); tree = cKDTree(rp)
    for s in (1.0, rr / cr):          # raw metric scale and scale-normalised
        q0 = cp * s; Uc = pca(q0)
        for sg in [(a, b, c) for a in (1, -1) for b in (1, -1) for c in (1, -1)]:
            Rm = Ur @ np.diag(sg) @ Uc.T
            if np.linalg.det(Rm) < 0: continue
            q = q0 @ Rm.T
            for _ in range(40):
                d, i = tree.query(q); Rm2, t2 = kab(rp[i], q); q = q @ Rm2.T + t2
            d1 = tree.query(q)[0]; d2 = cKDTree(q).query(rp)[0]
            r = dict(scale=float(s), chamfer_mm=float((d1.mean() + d2.mean()) / 2 * 1000), p90_mm=float(np.percentile(np.r_[d1, d2], 90) * 1000))
            if best is None or r['chamfer_mm'] < best['chamfer_mm']: best = r
    out.append(dict(label=lab, object=n, path=str(p), nv=len(v), nf=len(f), ext_cm=np.round(np.sort(np.ptp(v, 0)) * 100, 2).tolist(),
                    ref_ext_cm=np.round(np.sort(np.ptp(rv, 0)) * 100, 2).tolist(), **best))
    print(json.dumps(out[-1]), flush=True)
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
