"""Stereo table plane in OUR world per frame (non-board stereo points, RANSAC near z=0, inside the table disc): normal tilt vs world z
(deg), plane height at the table centre and at each board's reference centre (mm). usage: table_plane_check.py FRAMES OUT.json"""
import sys, json, numpy as np, cv2
sys.path.insert(0, __file__.rsplit('/', 1)[0]); import stereo_evidence as E
a, b, c = map(int, sys.argv[1].split(':')); Z = np.load(E.SRC / 'object_poses.npz'); Tw = Z['world_from_object']
ctr = np.array([0.04416647792665407, 0.3073668237396147]); rad = 0.226; rows = []
for k in range(a, min(b, 356), c):
    pts, uv = E.stereo_points(k); pw = pts @ E.W_rgb[k][:3, :3].T + E.W_rgb[k][:3, 3]; M = E.masks(k)
    ui = np.clip(np.rint(uv).astype(int), 0, [2027, 1519]); anym = np.zeros(len(pw), bool)
    for n in E.NAMES:
        if M[n] is not None: anym |= cv2.dilate(M[n].astype(np.uint8), np.ones((25, 25), np.uint8))[ui[:, 1], ui[:, 0]] > 0
    sel = (~anym) & (np.linalg.norm(pw[:, :2] - ctr, axis=1) < rad - .02) & (np.abs(pw[:, 2]) < .04); Q = pw[sel]
    rng = np.random.default_rng(0); best = None
    for _ in range(300):
        s = Q[rng.choice(len(Q), 3, replace=False)]; nrm = np.cross(s[1] - s[0], s[2] - s[0]); nn = np.linalg.norm(nrm)
        if nn < 1e-9: continue
        nrm /= nn; d = np.abs((Q - s[0]) @ nrm); inl = d < .003
        if best is None or inl.sum() > best[0]: best = (inl.sum(), inl)
    I = Q[best[1]]; A = np.c_[I[:, 0], I[:, 1], np.ones(len(I))]; coef = np.linalg.lstsq(A, I[:, 2], rcond=None)[0]   # z = ax + by + c
    nrm = np.array([-coef[0], -coef[1], 1]); nrm /= np.linalg.norm(nrm); tilt = float(np.degrees(np.arccos(nrm[2])))
    h = lambda xy: float((coef[0] * xy[0] + coef[1] * xy[1] + coef[2]) * 1000)
    row = dict(frame=k, n=int(len(Q)), inliers=int(best[0]), tilt_deg=tilt, coef=coef.tolist(), z_centre_mm=h(ctr), resid_mm=float(np.std(I[:, 2] - A @ coef) * 1000))
    for i, n in enumerate(E.NAMES):
        cc = Tw[k, i, :3, :3] @ E.samp[n].mean(0) + Tw[k, i, :3, 3]; row[n + '_z_mm'] = h(cc[:2])
    rows.append(row); print(f"{k/20:5.2f} n{row['n']:6d} inl{row['inliers']:6d} tilt {tilt:4.2f} zc {row['z_centre_mm']:5.1f} b1 {row['wooden_piece_1_z_mm']:5.1f} b2 {row['wooden_piece_2_z_mm']:5.1f} res {row['resid_mm']:.1f}", flush=True)
json.dump(rows, open(sys.argv[2], 'w'), indent=0)
