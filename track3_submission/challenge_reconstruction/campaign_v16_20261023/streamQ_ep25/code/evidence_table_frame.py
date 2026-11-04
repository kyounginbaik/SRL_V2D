"""Table-frame stereo evidence (the frame the physics and the official frame-0 anchor alignment care about). Per frame: RANSAC table
plane of the stereo points around the boards (board masks excluded), rigid map A taking that plane to z=0 (minimal rotation about the
inlier centroid + vertical shift). Then for each pose set: median point-to-surface distance and inlier fraction (< 8 mm) of the
A-mapped board-mask points, board face tilt vs z (pose) and vs the stereo board-face plane (flat boards), and top-edge height
(pose slab top vs stereo p98). usage: evidence_table_frame.py OUT.json FRAMES NAME=poses.npz [...]"""
import sys, json, numpy as np, cv2
sys.path.insert(0, __file__.rsplit('/', 1)[0]); import stereo_evidence as E
from scipy.spatial.transform import Rotation as R
SL = {}
for n in E.NAMES:
    v = np.asarray(E.mesh[n].vertices); c0 = v.mean(0); w, U = np.linalg.eigh((v - c0).T @ (v - c0))
    if np.linalg.det(U) < 0: U[:, 0] *= -1
    q = (v - c0) @ U; h, e = np.histogram(q[:, 0], bins=80); pk = np.sort(e[np.argsort(h)[-2:]] + np.diff(e)[0] / 2)
    lo = np.percentile(q, .5, 0); hi = np.percentile(q, 99.5, 0)
    SL[n] = dict(U=U, corners=c0 + np.array([[a, b, c] for a in pk for b in (lo[1], hi[1]) for c in (lo[2], hi[2])]) @ U.T)
out = sys.argv[1]; a, b, c = map(int, sys.argv[2].split(':')); poses = {s.split('=', 1)[0]: np.load(s.split('=', 1)[1])['world_from_object'] for s in sys.argv[3:]}
rng = np.random.default_rng(0); rows = []
for k in range(a, min(b, 356), c):
    pts, uv = E.stereo_points(k); pw = pts @ E.W_rgb[k][:3, :3].T + E.W_rgb[k][:3, 3]; M = E.masks(k); ui = np.clip(np.rint(uv).astype(int), 0, [2027, 1519])
    um = np.zeros((1520, 2028), np.uint8)
    for n in E.NAMES: um |= M[n].astype(np.uint8)
    near = cv2.dilate(um, np.ones((301, 301), np.uint8)) - cv2.dilate(um, np.ones((31, 31), np.uint8))
    Q = pw[(near[ui[:, 1], ui[:, 0]] > 0) & (np.abs(pw[:, 2]) < .05)]; best = None
    for _ in range(300):
        s = Q[rng.choice(len(Q), 3, replace=False)]; nr = np.cross(s[1] - s[0], s[2] - s[0]); nn = np.linalg.norm(nr)
        if nn < 1e-9: continue
        nr /= nn; inl = np.abs((Q - s[0]) @ nr) < .003
        if best is None or inl.sum() > best[0]: best = (inl.sum(), inl)
    I = Q[best[1]]; p0 = I.mean(0); _, _, Vt = np.linalg.svd(I - p0); nr = Vt[2] * np.sign(Vt[2][2])
    ax = np.cross(nr, [0, 0, 1.]); s_ = np.linalg.norm(ax); Ra = R.from_rotvec(ax / s_ * np.arctan2(s_, nr[2])).as_matrix() if s_ > 1e-12 else np.eye(3)
    pa = (pw - p0) @ Ra.T + np.array([p0[0], p0[1], 0.])
    row = dict(frame=k, t=k / 20, table_z_mm=float(p0[2] * 1000), table_tilt_deg=float(np.degrees(np.arccos(nr[2])))); line = f'{k/20:5.2f} tbl {p0[2]*1000:5.1f}/{row["table_tilt_deg"]:.1f}'
    for i, n in enumerate(E.NAMES):
        me = cv2.erode(M[n].astype(np.uint8), np.ones((7, 7), np.uint8)) > 0; B = pa[me[ui[:, 1], ui[:, 0]]]
        if len(B) < 50: continue
        Bs = B[rng.choice(len(B), min(4000, len(B)), replace=False)]; r = dict(n=int(len(B)), top_p98_mm=float(np.percentile(B[:, 2], 98) * 1000))
        Bf = B[np.abs(B[:, 2] - np.median(B[:, 2])) < .006]
        if np.ptp(B[:, 2]) < .03 and len(Bf) > 200:      # flat board: stereo face plane tilt
            _, _, V = np.linalg.svd(Bf - Bf.mean(0)); r['stereo_face_tilt_deg'] = float(np.degrees(np.arccos(abs(V[2][2]))))
        line += f' | {n[-1]} top{r["top_p98_mm"]:6.1f}' + (f' ftilt {r["stereo_face_tilt_deg"]:.1f}' if 'stereo_face_tilt_deg' in r else '')
        for nm, T in poses.items():
            Tw = T[k, i]; qb = (Bs - Tw[:3, 3]) @ Tw[:3, :3]; d, _ = E.tree[n].query(qb)
            C = SL[n]['corners'] @ Tw[:3, :3].T + Tw[:3, 3]; thin = Tw[:3, :3] @ SL[n]['U'][:, 0]
            r[nm] = dict(median_mm=float(np.median(d) * 1000), inlier8=float((d < .008).mean()), top_mm=float(C[:, 2].max() * 1000), bottom_mm=float(C[:, 2].min() * 1000),
                         face_tilt_deg=float(np.degrees(np.arccos(abs(thin[2])))))
            line += f' {nm}:{r[nm]["median_mm"]:4.1f}/{r[nm]["inlier8"]:.2f} top{r[nm]["top_mm"]:6.1f}' + (f' ft{r[nm]["face_tilt_deg"]:.1f}' if 'stereo_face_tilt_deg' in r else '')
        row[n] = r
    rows.append(row); print(line, flush=True)
json.dump(dict(poses={s.split('=', 1)[0]: s.split('=', 1)[1] for s in sys.argv[3:]}, rows=rows), open(out, 'w'), indent=0)
