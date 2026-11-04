"""R33: compare the teammate's CD-O tuning meshes with the meshes our ep33 package/target uses (shape only, GT-free): vertex/face count,
PCA extents, centroid offset from the mesh origin, long-axis direction and bowl end (spoon), symmetry axis and open end (cup), and the
residual after the best rigid ICP alignment (with and without uniform scale). usage: r33meshcmp.py NEW_DIR"""
import sys, numpy as np, trimesh
from pathlib import Path
N = Path(sys.argv[1]); R = Path('/data1/hansol/retarget33')
G18 = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates/G18_ep33_H13b1/episode_000033/meshes/episode_000033')
TG = R / 'targets/R33_HGA_r6/episode_000033'
def desc(v, lab):
    c = v.mean(0); w, U = np.linalg.eigh(np.cov((v - c).T)); p = (v - c) @ U
    print(f'  {lab:28s} n_vert {len(v):6d}  PCA extents cm {np.round(100*(p.max(0)-p.min(0)),2).tolist()}  centroid cm {np.round(100*c,2).tolist()}  long axis {np.round(U[:,2],3).tolist()}')
    return c, U
for obj in ('wooden_spoon', 'blue_cup'):
    print(f'\n== {obj}')
    meshes = {'new (CD-O)': N / f'{obj}.ply', 'G18 scene mesh': G18 / f'{obj}.obj', 'target mesh_metric': TG / f'{obj}_mesh_metric.obj'}
    V = {}
    for k, p in meshes.items():
        if not p.exists(): print('  missing', p); continue
        m = trimesh.load(p, force='mesh', process=False); V[k] = (m, np.asarray(m.vertices, float)); desc(V[k][1], k)
    if 'new (CD-O)' in V and 'G18 scene mesh' in V:
        from scipy.spatial import cKDTree
        pn = trimesh.sample.sample_surface(V['new (CD-O)'][0], 6000, seed=1)[0]; po = trimesh.sample.sample_surface(V['G18 scene mesh'][0], 6000, seed=2)[0]
        to = cKDTree(po)
        def icp(A, init, scale):
            M = init.copy()
            for _ in range(80):
                X = A @ M[:3, :3].T + M[:3, 3]; d, j = to.query(X); B = po[j]
                ca, cb = X.mean(0), B.mean(0); H = (X - ca).T @ (B - cb); U, S, Vt = np.linalg.svd(H); Rr = Vt.T @ U.T
                if np.linalg.det(Rr) < 0: Vt[-1] *= -1; Rr = Vt.T @ U.T
                sc = (S.sum() / ((X - ca) ** 2).sum()) if scale else 1.0
                D = np.eye(4); D[:3, :3] = sc * Rr; D[:3, 3] = cb - sc * Rr @ ca; M = D @ M
            X = A @ M[:3, :3].T + M[:3, 3]; d, _ = to.query(X); d2, _ = cKDTree(X).query(po)
            return M, np.concatenate([d, d2])
        for sc in (False, True):
            best = None
            for sgn in ([1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]):
                cn, Un = pn.mean(0), np.linalg.eigh(np.cov((pn-pn.mean(0)).T))[1]; co, Uo = po.mean(0), np.linalg.eigh(np.cov((po-po.mean(0)).T))[1]
                Ri = Uo @ np.diag(sgn) @ Un.T
                if np.linalg.det(Ri) < 0: Ri = Uo @ np.diag(np.array(sgn) * [1, 1, -1]) @ Un.T
                I = np.eye(4); I[:3, :3] = Ri; I[:3, 3] = co - Ri @ cn
                M, dd = icp(pn, I, sc)
                if best is None or dd.mean() < best[1].mean(): best = (M, dd)
            for init_name, I in (('identity', np.eye(4)),):
                M, dd = icp(pn, I, sc)
                if dd.mean() < best[1].mean(): best = (M, dd)
            M, dd = best; s_ = np.cbrt(abs(np.linalg.det(M[:3, :3]))); Rr = M[:3, :3] / s_
            ang = np.degrees(np.arccos(np.clip((np.trace(Rr) - 1) / 2, -1, 1)))
            print(f'  ICP new->G18 (scale={sc}): symmetric mean surface dist {100*dd.mean():.2f} cm, p95 {100*np.percentile(dd,95):.2f} cm | rotation {ang:.1f} deg, translation {np.round(100*M[:3,3],2).tolist()} cm, scale {s_:.3f}')
        d0, _ = to.query(pn); d1, _ = cKDTree(pn).query(po); dd0 = np.concatenate([d0, d1])
        print(f'  as given (no alignment): symmetric mean surface dist {100*dd0.mean():.2f} cm, p95 {100*np.percentile(dd0,95):.2f} cm')
