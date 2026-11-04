"""Frames where boards lie flat: board top-face height above the LOCAL stereo table (mapping-independent: both from the same stereo
cloud) vs the reference: reference top face / bottom face above world z=0 (the Compose table plane). Also each board's face tilt.
usage: flat_board_check.py FRAMES"""
import sys, numpy as np, cv2
sys.path.insert(0, __file__.rsplit('/', 1)[0]); import stereo_evidence as E
Z = np.load(E.SRC / 'object_poses.npz'); Tw = Z['world_from_object']
info = {}
for n in E.NAMES:
    v = np.asarray(E.mesh[n].vertices); c0 = v.mean(0); w, U = np.linalg.eigh((v - c0).T @ (v - c0))
    q = (v - c0) @ U; h, e = np.histogram(q[:, 0], bins=80); pk = np.sort(e[np.argsort(h)[-2:]] + np.diff(e)[0] / 2)
    info[n] = (c0, U, pk); print(n, 'thin range cm', q[:, 0].min() * 100, q[:, 0].max() * 100, 'face planes cm', pk * 100)
a, b, c = map(int, sys.argv[1].split(':'))
for k in range(a, b, c):
    pts, uv = E.stereo_points(k); pw = pts @ E.W_rgb[k][:3, :3].T + E.W_rgb[k][:3, 3]; M = E.masks(k); ui = np.clip(np.rint(uv).astype(int), 0, [2027, 1519])
    s = f'{k/20:5.2f}'
    for i, n in enumerate(E.NAMES):
        me = cv2.erode(M[n].astype(np.uint8), np.ones((15, 15), np.uint8)) > 0; sel = me[ui[:, 1], ui[:, 0]]; B = pw[sel]
        ring = cv2.dilate(M[n].astype(np.uint8), np.ones((61, 61), np.uint8)) - cv2.dilate(M[n].astype(np.uint8), np.ones((25, 25), np.uint8))
        oth = M[E.NAMES[1 - i]]; rs = (ring[ui[:, 1], ui[:, 0]] > 0) & ~oth[ui[:, 1], ui[:, 0]]; Rg = pw[rs]; Rg = Rg[np.abs(Rg[:, 2] - np.median(Rg[:, 2])) < .01]
        # local table plane from the ring, board points height above it
        A = np.c_[Rg[:, 0], Rg[:, 1], np.ones(len(Rg))]; cf = np.linalg.lstsq(A, Rg[:, 2], rcond=None)[0]
        hb = B[:, 2] - (cf[0] * B[:, 0] + cf[1] * B[:, 1] + cf[2]); hb = hb[(hb > -.01) & (hb < .012)]   # top face (exclude marker balls)
        # board face tilt from stereo (plane fit to board points) vs local table
        Bb = B[np.abs(B[:, 2] - np.median(B[:, 2])) < .006]; Ab = np.c_[Bb[:, 0], Bb[:, 1], np.ones(len(Bb))]; cb = np.linalg.lstsq(Ab, Bb[:, 2], rcond=None)[0]
        nb = np.array([-cb[0], -cb[1], 1]); nt = np.array([-cf[0], -cf[1], 1]); tilt_st = np.degrees(np.arccos(nb @ nt / np.linalg.norm(nb) / np.linalg.norm(nt)))
        c0, U, pk = info[n]; Rw = Tw[k, i, :3, :3]; tw = Tw[k, i, :3, 3]; Fr = Rw @ U; thin = Fr[:, 0]
        cc = Rw @ c0 + tw; faces = [cc + thin * p for p in pk]; fz = sorted(f[2] for f in faces)
        tilt_ref = np.degrees(np.arccos(abs(thin[2])))
        s += f' | {n[-1]}: stereo top-above-local-table {np.median(hb)*1000:5.1f}mm tilt {tilt_st:4.1f} | ref bottom {fz[0]*1000:6.1f} top {fz[1]*1000:6.1f}mm (world z0) tilt {tilt_ref:4.1f} | local table in world {np.median(Rg[:,2])*1000:5.1f}'
    print(s, flush=True)
