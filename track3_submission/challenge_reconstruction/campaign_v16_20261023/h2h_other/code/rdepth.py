"""Mask-free model->data depth check: splat dense posed-surface samples of each track's mesh into the 1014x760 AnyCalib image
(z-buffer = nearest sample per pixel), compare with the stereo depth image depth_rgb (same image, project_depth.py) at those
pixels. Pixels where the stereo surface is > 2 cm IN FRONT of the model are treated as occluders (hands/other object) and
ignored; reported: median |dz| (mm) of the rest, fraction of model pixels with |dz| < 1 cm ('agree'), fraction where the stereo
is > 2 cm BEHIND the model ('model floats in free space' -> the model is where nothing is). usage: rdepth.py EP STRIDE OUT.json"""
import sys, os, json, numpy as np, cv2, trimesh
sys.path.insert(0, os.path.dirname(__file__)); from common import *
ep, stride, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
o = load_ours(ep); n = load_new(ep, o['names']); ev = Evidence(ep, o['names']); names = n['names']
SAMP = {}
for tn, ms in (('ours', o['meshes']), ('new', n['meshes'])):
    for k in names: SAMP[(tn, k)] = trimesh.sample.sample_surface(ms[k], 120000, seed=0)[0]
W, H = ev.size[0] // 2, ev.size[1] // 2; rows = []
T = min(len(o['A_T_obj']), *[len(n['A_T_obj'][k]) for k in names])
for t in range(0, T, stride):
    p = ev.sd / f'depth_rgb/{t:06d}.png'; dimg = cv2.imread(str(p), -1)
    if dimg is None: continue
    dep = np.where(dimg < 65535, 65535.0 / np.maximum(dimg, 1) - 1, np.nan); row = dict(frame=t)
    for k in names:
        r = {}
        for tn in ('ours', 'new'):
            A = o['A_T_obj'][t, o['names'].index(k)] if tn == 'ours' else n['A_T_obj'][k][t]
            P = SAMP[(tn, k)] @ A[:3, :3].T + A[:3, 3]; uv = ev.project(P) * 0.5; ok = np.isfinite(uv).all(1)
            u = np.round(uv[ok, 0]).astype(int); v = np.round(uv[ok, 1]).astype(int); z = P[ok, 2]; inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            zb = np.full(H * W, np.inf); np.minimum.at(zb, v[inb] * W + u[inb], z[inb]); zb = zb.reshape(H, W)
            m = np.isfinite(zb) & np.isfinite(dep) & (dep > 0.1)
            dz = dep[m] - zb[m]; occl = dz < -0.02; rest = dz[~occl]
            if len(rest) < 50: r[tn] = {}; continue
            r[tn] = dict(n=int(m.sum()), occl=float(occl.mean()), med_mm=float(np.median(np.abs(rest)) * 1000), agree=float((np.abs(rest) < .01).mean()), behind=float((rest > .02).mean()))
        row[k] = r
    rows.append(row)
    print(t, ' | '.join(f"{k[:10]} " + ' '.join(f"{tn}:{row[k][tn].get('med_mm',np.nan):.1f}/{row[k][tn].get('agree',np.nan):.2f}/{row[k][tn].get('behind',np.nan):.2f}" for tn in ('ours', 'new')) for k in names), flush=True)
json.dump(dict(ep=ep, names=names, rows=rows), open(out, 'w'))
