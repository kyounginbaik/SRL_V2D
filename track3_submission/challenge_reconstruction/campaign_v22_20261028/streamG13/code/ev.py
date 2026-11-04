"""GT-free image evidence per frame/object for one hypothesis (no reference data): silhouette IoU vs SAM2 mask (hands/other objects
excluded as in qc.py), SIGNED median depth residual (mesh z - stereo z, mm; + = mesh behind stereo), |residual|, and the 2D centroid offset
(rendered - mask, px) on non-excluded pixels. Half resolution (507x380), z-buffered surface splats.
usage: ev.py EPROOT HYP.npz OUT.json     HYP.npz: camera_from_object [F,n,4,4], names [n], meshes [n] (paths)"""
import sys, json, numpy as np, cv2, trimesh
root, hyp, out = sys.argv[1:4]
H = np.load(hyp, allow_pickle=True); C = H['camera_from_object']; names = [str(n) for n in H['names']]; meshes = [str(m) for m in H['meshes']]
intr = json.load(open(f'{root}/intrinsics.json')); W, Hh = intr['width'] // 2, intr['height'] // 2
K = np.array([[intr['fx'] / 2, 0, intr['cx'] / 2], [0, intr['fy'] / 2, intr['cy'] / 2], [0, 0, 1.]])
rd = lambda p: (cv2.resize(v, (W, Hh), interpolation=cv2.INTER_NEAREST) > 0) if (v := cv2.imread(p, 0)) is not None else None
P = []; TRI = []
for m in meshes:
    mm = trimesh.load(m, force='mesh', process=False); p, fi = trimesh.sample.sample_surface(mm, 120000, seed=7); P.append((p, mm.face_normals[fi]))
    TRI.append((np.asarray(mm.vertices), np.asarray(mm.faces)))
F = C.shape[0] if len(sys.argv) < 5 else int(sys.argv[4]); res = {n: dict(iou=[], dz=[], adz=[], dcx=[], dcy=[], npx=[]) for n in names}
for f in range(F):
    enc = cv2.resize(cv2.imread(f'{root}/depth/{f:06d}.png', -1), (W, Hh), interpolation=cv2.INTER_NEAREST)
    depth = 65535 / np.maximum(enc.astype(float), 1) - 1; depth[(enc == 0) | (enc >= 65535)] = np.nan
    masks = {n: rd(f'{root}/objects/{n}/masks/1/{f:06d}.png') for n in names}
    for s, n in enumerate(names):
        r = res[n]; mask = masks[n]
        hands = [v for l in '23' if (v := rd(f'{root}/objects/{n}/masks/{l}/{f:06d}.png')) is not None]
        others = [masks[o] for o in names if o != n and masks[o] is not None]
        if mask is None or mask.sum() < 30:
            for k in r: r[k].append(None)
            continue
        excl = np.zeros((Hh, W), bool)
        for v in hands + others: excl |= v
        excl = cv2.dilate(excl.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        keep = mask.copy()
        for v in others:
            if v.sum() < mask.sum(): keep &= ~v
        excl &= ~keep
        T = C[f, s]; p, nn = P[s]; c = p @ T[:3, :3].T + T[:3, 3]; fac = (np.einsum('ij,ij->i', nn @ T[:3, :3].T, c) < 0) & (c[:, 2] > .05); c = c[fac]
        uv = np.rint(c[:, :2] / c[:, 2:] @ K[:2, :2].T + K[:2, 2]).astype(int); ins = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < Hh)
        uv, c = uv[ins], c[ins]
        z = np.full(W * Hh, np.inf); np.minimum.at(z, uv[:, 1] * W + uv[:, 0], c[:, 2]); z = z.reshape(Hh, W); pred = np.isfinite(z)
        vv, ff = TRI[s]; cw = vv @ T[:3, :3].T + T[:3, 3]; okf = (cw[ff][:, :, 2] > .05).all(1)
        pv = cw[:, :2] / np.maximum(cw[:, 2:], 1e-3) @ K[:2, :2].T + K[:2, 2]
        sil = np.zeros((Hh, W), np.uint8); cv2.fillPoly(sil, [t for t in np.rint(pv[ff[okf]] * 4).astype(np.int32)], 1, lineType=cv2.LINE_8, shift=2)
        reg = sil > 0
        U = (reg | mask) & ~excl; O = reg & mask & ~excl
        r['iou'].append(float(O.sum() / max(1, U.sum())))
        us = pred & mask & ~excl & np.isfinite(depth); d = z[us] - depth[us]
        r['dz'].append(float(np.median(d) * 1000) if len(d) >= 12 else None); r['adz'].append(float(np.median(np.abs(d)) * 1000) if len(d) >= 12 else None)
        a = reg & ~excl; b = mask & ~excl
        if a.sum() > 20 and b.sum() > 20:
            ya, xa = np.nonzero(a); yb, xb = np.nonzero(b); r['dcx'].append(float(xa.mean() - xb.mean()) * 2); r['dcy'].append(float(ya.mean() - yb.mean()) * 2)
        else: r['dcx'].append(None); r['dcy'].append(None)
        r['npx'].append(int(b.sum()))
json.dump(dict(hyp=hyp, names=names, per_frame=res), open(out, 'w'))
def m(x): x = [v for v in x if v is not None]; return round(float(np.mean(x)), 3) if x else None
for n in names: print(n, {k: m(v) for k, v in res[n].items()})
