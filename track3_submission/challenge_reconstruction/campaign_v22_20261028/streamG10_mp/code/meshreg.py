"""Scorer-side diagnostic (no output enters any bundle): how OUR object mesh registers (kit rule: scale-normalised ICP) onto a
reference mesh (compose = scan-template; or public scan for public clips). Reports extents, gyration radii, kit scale,
chamfer after registration, and the anchor displacement: where the reference's surface centroid lands on our mesh vs
our own surface centroid (a constant body-frame anchor offset = constant position error in every frame).
usage: meshreg.py OURS_OBJ REF_OBJ EP SLOT"""
import sys, numpy as np
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG10_mp/code')
from g10lib import M, loadmesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
a, b, ep, slot = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
cv, cf = loadmesh(a, 1 << 30); rv, rf = loadmesh(b)
cc, cr = M._surface_moments(cv, cf); rc, rr = M._surface_moments(rv, rf)
def ext(v):
    x = v - v.mean(0); w, U = np.linalg.eigh(x.T @ x); p = x @ U; return np.ptp(p, 0)
print('ours extents cm', np.round(ext(cv) * 100, 1), 'gyr', round(cr * 100, 2), '| ref extents', np.round(ext(rv) * 100, 1), 'gyr', round(rr * 100, 2), 'scale', round(rr / cr, 3))
res = []
for s in range(5):
    seed = 1_000_003 * ep + 101 * slot + s
    ca, R = M._register_object_body((cv, cf), (rv, rf), rc, seed)
    # map our vertices into the reference frame: p_ref = R (p - cc) * scale + rc + offset   (offset implied by anchor relation)
    cp = M._sample_mesh_surface(cv, cf, 4000, seed); rp = M._sample_mesh_surface(rv, rf, 4000, seed)
    sc = rr / cr
    # candidate_anchor = cc + R^T (anchor - rc - offset)/scale with anchor = rc -> offset = -R (ca - cc) * scale
    off = -R @ (ca - cc) * sc
    mp = ((cp - cc) * sc) @ R.T + rc + off
    d1 = cKDTree(rp).query(mp)[0]; d2 = cKDTree(mp).query(rp)[0]
    res.append((seed % 1000, round(float(np.degrees(Rot.from_matrix(R).magnitude())), 1), np.round((ca - cc) * 100, 2).tolist(), round(float(np.linalg.norm(ca - cc) * 100), 2), round(float(0.5 * (d1.mean() + d2.mean()) * 100), 3)))
for r in res: print(' seed', r[0], 'R deg', r[1], 'anchor-on-ours minus our centroid (cm)', r[2], '|', r[3], 'chamfer cm (scaled)', r[4])
