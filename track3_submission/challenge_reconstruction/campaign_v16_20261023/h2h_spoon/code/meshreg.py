"""Register OUR mesh (body frame) to the NEW mesh (body frame): G with p_new = G @ p_ours. PCA init over the 4 proper sign
combos + point-to-point ICP, pick the lowest symmetric Chamfer. Also reports the 2nd best (e.g. 180-deg roll twin of the spoon)."""
import numpy as np, trimesh
from scipy.spatial import cKDTree
def _samp(m, n=8000, seed=0):
    return np.asarray(trimesh.sample.sample_surface_even(m, n, seed=seed)[0]) if hasattr(trimesh.sample, 'sample_surface_even') else m.sample(n)
def icp(A, B, T, it=60):
    tb = cKDTree(B)
    for _ in range(it):
        X = A @ T[:3, :3].T + T[:3, 3]; d, j = tb.query(X); keep = d < np.percentile(d, 90)
        P, Q = X[keep], B[j[keep]]; mp, mq = P.mean(0), Q.mean(0); U, S, Vt = np.linalg.svd((P - mp).T @ (Q - mq))
        Rr = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T; dT = np.eye(4); dT[:3, :3] = Rr; dT[:3, 3] = mq - Rr @ mp; T = dT @ T
    X = A @ T[:3, :3].T + T[:3, 3]; c = (cKDTree(B).query(X)[0].mean() + cKDTree(X).query(B)[0].mean()) / 2
    return T, c
def register(ma, mb):
    A, B = _samp(ma), _samp(mb, seed=1); ca, cb = A.mean(0), B.mean(0)
    ea = np.linalg.eigh(np.cov((A - ca).T))[1]; eb = np.linalg.eigh(np.cov((B - cb).T))[1]; res = []
    for s in [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]:
        Rr = eb @ np.diag(s) @ ea.T
        if np.linalg.det(Rr) < 0: Rr = eb @ np.diag(s) @ np.diag([1, 1, -1]) @ ea.T
        T = np.eye(4); T[:3, :3] = Rr; T[:3, 3] = cb - Rr @ ca; res.append(icp(A, B, T))
    # also identity init
    res.append(icp(A, B, np.eye(4)))
    res.sort(key=lambda x: x[1]); return res
