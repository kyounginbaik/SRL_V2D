"""Camera-path check against the static background: back-project the stereo depth of non-object, non-hand pixels (0.25-1.2 m),
map to the world with each track's camera path (ours: world_from_rgb, Kout rays; new: cam0_T_cam conjugated to cam a, official
rays), and measure the median nearest-neighbour distance (mm) to the frame-REF background cloud. A correct path keeps the static
scene static. usage: campath.py EP OUT.json [STEP]"""
import sys, json, numpy as np, cv2
sys.path.insert(0, __file__.rsplit('/', 1)[0]); import h2hlib as H
from scipy.spatial import cKDTree
ep, out = int(sys.argv[1]), sys.argv[2]; step = int(sys.argv[3]) if len(sys.argv) > 3 else 5
e = H.Ep(ep, nsamp=10); mo = {t: e.model_of(t) for t in ('ours', 'new')}
camw = {'ours': lambda f: e.wfc[f], 'new': lambda f: np.linalg.inv(e.Ta) @ e.c0Tc[f] @ e.Ta @ np.linalg.inv(e.Ta)}
# new camera a in their world (rectL frame 0): world_from_a = c0Tc[f] @ inv(Ta)
camw['new'] = lambda f: e.c0Tc[f] @ np.linalg.inv(e.Ta)
def cloud(tr, f):
    dep = e.depth(f); ex = e.hand(f)
    for o in e.names: ex |= e.mask(o, f)
    ex = cv2.dilate(ex.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    sel = ~ex & np.isfinite(dep) & (dep > .25) & (dep < 2.0); sel[::2] = False; sel[:, ::2] = False
    P = e.cm.backproject(mo[tr], sel, dep); W = camw[tr](f); return P @ W[:3, :3].T + W[:3, 3]
import meshreg
def icp_corr(P, tree, Q, it=30):
    T = np.eye(4)
    for _ in range(it):
        X = P @ T[:3, :3].T + T[:3, 3]; d, j = tree.query(X, distance_upper_bound=0.05); k = np.isfinite(d)
        if k.sum() < 200: return None
        A, B = X[k], Q[j[k]]; ma, mb = A.mean(0), B.mean(0); U, S_, Vt = np.linalg.svd((A - ma).T @ (B - mb))
        Rr = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T; dT = np.eye(4); dT[:3, :3] = Rr; dT[:3, 3] = mb - Rr @ ma; T = dT @ T
    return T
rows = []; refs = {}; refP = {}
for ref in (0, e.N // 2):
    for tr in ('ours', 'new'): refP[(tr, ref)] = cloud(tr, ref); refs[(tr, ref)] = cKDTree(refP[(tr, ref)])
for f in range(0, e.N, step):
    r = dict(frame=f)
    for tr in ('ours', 'new'):
        P = cloud(tr, f)
        for ref in (0, e.N // 2):
            d = refs[(tr, ref)].query(P, distance_upper_bound=0.2)[0]; d = d[np.isfinite(d)]
            r[f'{tr}_ref{ref}'] = float(np.median(d) * 1000) if len(d) > 100 else None
            if ref == 0:
                T = icp_corr(P[::3], refs[(tr, ref)], refP[(tr, ref)])
                r[f'{tr}_icp_cm'] = float(np.linalg.norm(T[:3, 3]) * 100) if T is not None else None
                r[f'{tr}_icp_deg'] = float(np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1)))) if T is not None else None
    rows.append(r); print(f, ' '.join(f'{k}={v:.1f}' for k, v in r.items() if k != 'frame' and v is not None), flush=True)
json.dump(rows, open(out, 'w'))
