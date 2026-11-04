"""Mapping-independent board height evidence. Per frame: local table plane = RANSAC plane of stereo points in a ring around both boards
(excluding board masks), then for each board the 98th / 99th percentile height of its (eroded) mask points above that plane = measured
top-edge / top-face height. Reference: the highest slab corner (6 mm face planes, feet/markers excluded) above the Compose table z=0,
and the lowest slab corner (bottom). If a board rests on the table the reference top must equal the measured top.
usage: height_check.py FRAMES OUT.json NAME=object_poses.npz [...]"""
import sys, json, numpy as np, cv2
sys.path.insert(0, __file__.rsplit('/', 1)[0]); import stereo_evidence as E
SL = {}
for n in E.NAMES:
    v = np.asarray(E.mesh[n].vertices); c0 = v.mean(0); w, U = np.linalg.eigh((v - c0).T @ (v - c0))
    if np.linalg.det(U) < 0: U[:, 0] *= -1
    q = (v - c0) @ U; h, e = np.histogram(q[:, 0], bins=80); pk = np.sort(e[np.argsort(h)[-2:]] + np.diff(e)[0] / 2)
    lo = np.percentile(q, .5, 0); hi = np.percentile(q, 99.5, 0)
    corners = np.array([[a, b, c] for a in pk for b in (lo[1], hi[1]) for c in (lo[2], hi[2])]); SL[n] = c0 + corners @ U.T
a, b, c = map(int, sys.argv[1].split(':')); poses = {s.split('=', 1)[0]: np.load(s.split('=', 1)[1])['world_from_object'] for s in sys.argv[3:]}
rows = []; rng = np.random.default_rng(0)
for k in range(a, min(b, 356), c):
    pts, uv = E.stereo_points(k); pw = pts @ E.W_rgb[k][:3, :3].T + E.W_rgb[k][:3, 3]; M = E.masks(k); ui = np.clip(np.rint(uv).astype(int), 0, [2027, 1519])
    um = np.zeros((1520, 2028), np.uint8)
    for n in E.NAMES: um |= M[n].astype(np.uint8)
    near = cv2.dilate(um, np.ones((301, 301), np.uint8)) - cv2.dilate(um, np.ones((31, 31), np.uint8))
    rs = near[ui[:, 1], ui[:, 0]] > 0; Q = pw[rs & (np.abs(pw[:, 2]) < .05)]
    best = None
    for _ in range(300):
        s = Q[rng.choice(len(Q), 3, replace=False)]; nr = np.cross(s[1] - s[0], s[2] - s[0]); nn = np.linalg.norm(nr)
        if nn < 1e-9: continue
        nr /= nn; inl = np.abs((Q - s[0]) @ nr) < .003
        if best is None or inl.sum() > best[0]: best = (inl.sum(), inl, nr if nr[2] > 0 else -nr, s[0])
    I = Q[best[1]]; nr = best[2]; p0 = I.mean(0)
    row = dict(frame=k, t=k / 20, table_inliers=int(best[0]), table_normal_tilt_deg=float(np.degrees(np.arccos(nr[2]))), table_z_mm=float(p0[2] * 1000))
    s = f'{k/20:5.2f} tbl {p0[2]*1000:5.1f}mm/{row["table_normal_tilt_deg"]:.1f}deg'
    for i, n in enumerate(E.NAMES):
        me = cv2.erode(M[n].astype(np.uint8), np.ones((9, 9), np.uint8)) > 0; B = pw[me[ui[:, 1], ui[:, 0]]]
        if len(B) < 50: s += f' | {n[-1]} -'; continue
        hgt = (B - p0) @ nr; r = dict(n=int(len(B)), meas_p98_mm=float(np.percentile(hgt, 98) * 1000), meas_p995_mm=float(np.percentile(hgt, 99.5) * 1000), meas_p50_mm=float(np.median(hgt) * 1000))
        for nm, T in poses.items():
            C = SL[n] @ T[k, i, :3, :3].T + T[k, i, :3, 3]; r[nm] = dict(top_mm=float(C[:, 2].max() * 1000), bottom_mm=float(C[:, 2].min() * 1000))
        row[n] = r; s += f' | {n[-1]} meas p98 {r["meas_p98_mm"]:6.1f} ' + ' '.join(f'{nm} top {r[nm]["top_mm"]:6.1f} bot {r[nm]["bottom_mm"]:6.1f}' for nm in poses)
    rows.append(row); print(s, flush=True)
json.dump(rows, open(sys.argv[2], 'w'), indent=0)
