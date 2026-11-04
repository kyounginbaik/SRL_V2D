"""SCORER-SIDE DIAGNOSTIC ONLY (public GT read here; never used to fit anything that enters a reconstruction).
Question: at the start of each public clip, do our GT-free image cues (SAM2 silhouettes, stereo depth, table plane) prefer
the GT-implied pose of OUR mesh (GT pose mapped back through the kit's body registration and frame-0 scene alignment) over
our own pose? If not, a GT-free frame-0 refinement cannot see the relative-placement error.
Usage: gtdiag.py EP TAG -> out/gtdiag_<TAG>_<EP>.json"""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG15/code')
sys.path.insert(0, str(Path(__file__).parent))
import kitx
from g17lib import *
ep, tag = int(sys.argv[1]), sys.argv[2]
X = kitx.load_export(ep, tag); names = X['names']
o = kitx.score(ep, {n: X['W'][:, i] for i, n in enumerate(names)}, X['meshes'])
M = kitx.M; S, off = o['S'], o['off']; F = o['F']
split = 'public'; E = Episode(kitx.root(ep)); cam_bc = X['cam'].astype(float)
cam_rgb = shift(cam_bc, -np.asarray(delta_arr(split, ep, len(cam_bc))))
Wgt = {}
for s, n in enumerate(o['objs']):
    Rr = M._rotation_matrix(o['R'][:, s, 3:]); pr = o['R'][:, s, :3]
    pos = (pr - off) @ S          # S^T (p - off)
    Rreg = np.einsum('ji,tjk->tik', S, Rr)
    Ro = Rreg @ o['rots'][s]; po = pos - np.einsum('tij,j->ti', Ro, o['cas'][s])
    T = np.tile(np.eye(4), (F, 1, 1)); T[:, :3, :3] = Ro; T[:, :3, 3] = po; Wgt[n] = T
meshes = {n: Mesh(X['meshes'][n]) for n in names}
frames = list(range(0, min(F, 40), 4))
res = dict(ep=ep, tag=tag, objs={}, scorer_side_only=True)
nw, dw = E.plane_world(cam_bc, frames)
for n in o['objs']:
    i = names.index(n); r = {k: [] for k in ['iou_o', 'iou_g', 'cen_o', 'cen_g', 'dep_o', 'dep_g']}
    for f in frames:
        m = E.mask(n, 1, f); hands = E.hands(names, f); exc = hands.copy()
        for q in names:
            if q != n and E.mask(q, 1, f) is not None: exc |= E.mask(q, 1, f)
        exc &= ~m if m is not None else exc
        dep = E.depth(f)
        for lab, T in (('o', X['W'][f, i]), ('g', Wgt[n][f])):
            Tc = np.linalg.inv(cam_rgb[f]) @ T; sil = render_sil(E, meshes[n], Tc)
            if sil is None or m is None: continue
            u = (sil | m) & ~exc; r['iou_' + lab].append(float((sil & m & ~exc).sum() / max(1, u.sum())))
            ys, xs = np.nonzero(m & ~exc); ys2, xs2 = np.nonzero(sil & ~exc)
            if len(xs) > 30 and len(xs2) > 30: r['cen_' + lab].append([float(xs2.mean() - xs.mean()), float(ys2.mean() - ys.mean())])
            zr = render_depth(E, meshes[n], np.linalg.inv(cam_bc[f]) @ T)
            use = m & np.isfinite(zr) & np.isfinite(dep) & ~exc
            if use.sum() > 100: r['dep_' + lab].append(float(np.median(dep[use] - zr[use]) * 100))
    T0o, T0g = X['W'][0, i], Wgt[n][0]
    r['height_o'] = float(((meshes[n].Vh @ T0o[:3, :3].T + T0o[:3, 3]) @ nw + dw).min() * 100)
    r['height_g'] = float(((meshes[n].Vh @ T0g[:3, :3].T + T0g[:3, 3]) @ nw + dw).min() * 100)
    c0 = cam_rgb[0]; dvec = np.linalg.inv(c0)[:3, :3] @ (T0g[:3, 3] - T0o[:3, 3]) * 100
    r['f0_gt_minus_ours_cam_cm'] = dvec.tolist(); r['pos_err_mean'] = float(o['pos'][:, o['objs'].index(n)].mean()); r['f0pos'] = float(o['pos'][0, o['objs'].index(n)])
    res['objs'][n] = r
    print(f"{ep:3d} {n:18s} err {r['pos_err_mean']:.2f} f0 {r['f0pos']:.2f} | gt-ours cam {np.round(dvec,2)} | IoU o {np.mean(r['iou_o']):.3f} g {np.mean(r['iou_g']):.3f} | cen o {np.round(np.mean(r['cen_o'],0),1)} g {np.round(np.mean(r['cen_g'],0),1)} | dep o {np.mean(r['dep_o']):+.2f} g {np.mean(r['dep_g']):+.2f} | h o {r['height_o']:+.2f} g {r['height_g']:+.2f}", flush=True)
Path(__file__).parent.parent.joinpath('out', f'gtdiag_{tag}_{ep}.json').write_text(json.dumps(res, indent=1))
