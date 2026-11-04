"""G18 task 3 (scorer-side diagnostic ONLY; public scans read here, nothing written into any mesh/pose):
yes/no -- do the public GT scans contain the gray mocap balls? Detect ~6 mm spheres protruding from the surface:
sample the scan surface, RANSAC-fit spheres of radius 4-9 mm on local neighbourhoods, count well-supported, separated ones.
Output: counts and radii only (no positions are kept)."""
import os, sys, json, numpy as np, trimesh
from scipy.spatial import cKDTree
T3 = '/data1/video_to_data_challenge/track_3/public/mesh'
def fit_sphere(P):
    A = np.c_[2 * P, np.ones(len(P))]; b = (P ** 2).sum(1)
    x, *_ = np.linalg.lstsq(A, b, rcond=None); c = x[:3]; r = np.sqrt(max(x[3] + c @ c, 0)); return c, r
res = {}
rng = np.random.default_rng(0)
for o in sorted(os.listdir(T3)):
    fs = [f for f in os.listdir(f'{T3}/{o}') if f.endswith('.glb') and 'collision' not in f]
    m = trimesh.load(f'{T3}/{o}/{fs[0]}', force='mesh')
    P, _ = trimesh.sample.sample_surface(m, 60000, seed=0); P = np.asarray(P)
    tree = cKDTree(P); found = []
    for i in rng.choice(len(P), 4000, replace=False):
        nb = P[tree.query_ball_point(P[i], 0.009)]
        if len(nb) < 30: continue
        c, r = fit_sphere(nb)
        if not (0.004 < r < 0.009): continue
        resid = np.abs(np.linalg.norm(nb - c, axis=1) - r)
        if np.median(resid) > 0.0004: continue
        # sphere must be well covered (> 55% of its surface sampled -> a protruding ball, not a curved wall)
        sp = P[tree.query_ball_point(c, r + 0.0015)]; d = np.abs(np.linalg.norm(sp - c, axis=1) - r)
        sp = sp[d < 0.0008]
        if len(sp) < 20: continue
        u = (sp - c) / np.linalg.norm(sp - c, axis=1, keepdims=True)
        dirs = trimesh.creation.icosphere(2).vertices
        cov = (np.max(dirs @ u.T, axis=1) > 0.93).mean()
        if cov < 0.55: continue
        if all(np.linalg.norm(c - f[0]) > 0.012 for f in found): found.append((c, r, cov))
    res[o] = dict(n_balls=len(found), radii_mm=[round(f[1] * 1000, 1) for f in found], coverage=[round(f[2], 2) for f in found],
                  extents_cm=(np.asarray(m.extents) * 100).round(1).tolist())
    print(o, res[o], flush=True)
json.dump(res, open(sys.argv[1], 'w'), indent=1)
