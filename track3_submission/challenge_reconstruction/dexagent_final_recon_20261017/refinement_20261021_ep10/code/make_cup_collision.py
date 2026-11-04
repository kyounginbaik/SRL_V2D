"""Thin-wall cup collision from the scan: per height band, inner and outer radius = median over 36 angular bins of the per-bin
min / max radius (robust to the marker knobs); NA angular sectors x NB bands of convex frustum sectors (inner chord vertices
pushed out by 1/cos(pi/NA) so the chord is tangent to the true inner surface) + a base disk. Built in the scan frame and moved
into our body frame with the exact rigid transform of the placed visual mesh. Replaces <cup>_convex in the source copy and the
episode-dir stage2 assets (CoACD parts kept as <cup>_convex_coacd), rewrites URDF collision list + inertia, updates manifest."""
import json, shutil, hashlib, sys, numpy as np, trimesh
import xml.etree.ElementTree as ET
from pathlib import Path
EP10 = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep10')
S = EP10 / 'source/episode_000010'; E = EP10 / 'episode_dir/episode_000010'; NA, NB = 24, 10
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
rep = {}
for n in ('blue_cup', 'beige_cup'):
    scan = trimesh.load(f'/data1/SRL_V2D/gt_best_per_object/{n}.ply', force='mesh', process=False); X = np.asarray(scan.vertices, float)
    Y = np.asarray(trimesh.load(S / f'assets/{n}_visual.obj', force='mesh', process=False).vertices, float)
    cx, cy = X.mean(0), Y.mean(0); U, _, Vt = np.linalg.svd((X - cx).T @ (Y - cy)); D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]); Rm = Vt.T @ D @ U.T; t = cy - Rm @ cx
    assert np.abs(X @ Rm.T + t - Y).max() < 1e-6, 'visual is not a rigid copy of the scan'
    P = trimesh.sample.sample_surface(scan, 300000, seed=0)[0]; c = P.mean(0); w, Ue = np.linalg.eigh(np.cov((P - c).T)); a = Ue[:, -1]
    h = (P - c) @ a; rr = np.linalg.norm((P - c) - np.outer(h, a), axis=1)
    if rr[h > np.percentile(h, 85)].mean() < rr[h < np.percentile(h, 15)].mean(): a = -a; h = -h
    e1 = np.cross(a, [1, 0, 0]); e1 = e1 / np.linalg.norm(e1) if np.linalg.norm(e1) > .1 else np.cross(a, [0, 1, 0]) / np.linalg.norm(np.cross(a, [0, 1, 0])); e2 = np.cross(a, e1)
    ang = np.arctan2((P - c) @ e2, (P - c) @ e1); h0, h1 = h.min(), h.max()
    base_top = None; edges = np.linspace(h0, h1, NB + 1); prof = []
    # base thickness: lowest height where an inner cavity appears (inner radius clearly below outer)
    def radii(lo, hi):
        s = (h >= lo) & (h < hi); ins, outs = [], []
        for k in range(36):
            q = s & (ang >= -np.pi + k * 2 * np.pi / 36) & (ang < -np.pi + (k + 1) * 2 * np.pi / 36)
            if q.sum() > 5: ins.append(rr[q].min()); outs.append(rr[q].max())
        return (np.median(ins), np.median(outs)) if ins else (np.nan, np.nan)
    fine = np.linspace(h0, h0 + 0.03, 16)
    for lo, hi in zip(fine[:-1], fine[1:]):
        ri, ro = radii(lo, hi)
        if ro - ri < 0.006 and ri > 0.01: base_top = lo; break
    base_top = base_top if base_top is not None else h0 + 0.004
    edges = np.linspace(base_top, h1, NB + 1); R_in, R_out = [], []
    for z in edges:
        ri, ro = radii(z - 0.004, z + 0.004); R_in.append(ri); R_out.append(ro)
    R_in, R_out = np.array(R_in), np.array(R_out); thick = np.clip(R_out - R_in, 0.0015, 0.006); R_in = R_out - thick
    parts = []; da = 2 * np.pi / NA; push = 1 / np.cos(np.pi / NA)
    for b in range(NB):
        for k in range(NA):
            pts = []
            for z, ri, ro in ((edges[b], R_in[b], R_out[b]), (edges[b + 1], R_in[b + 1], R_out[b + 1])):
                for th in (k * da, (k + 1) * da):
                    d = np.cos(th) * e1 + np.sin(th) * e2
                    pts += [c + a * z + d * ri * push, c + a * z + d * ro * push]
            parts.append(trimesh.convex.convex_hull(np.array(pts)))
    base = []
    for th in np.linspace(0, 2 * np.pi, 33)[:-1]:
        d = np.cos(th) * e1 + np.sin(th) * e2; rb = radii(h0, base_top)[1] if np.isfinite(radii(h0, base_top)[1]) else R_out[0]
        base += [c + a * h0 + d * rb, c + a * base_top + d * R_out[0]]
    parts.append(trimesh.convex.convex_hull(np.array(base)))
    parts = [trimesh.Trimesh(p.vertices @ Rm.T + t, p.faces) for p in parts]   # scan frame -> our body frame
    vol = np.array([p.volume for p in parts]); rho = 0.1 / vol.sum(); com = sum(p.volume * p.center_mass for p in parts) / vol.sum(); I = np.zeros((3, 3))
    for p in parts:
        m_i = rho * p.volume; d = p.center_mass - com; I += p.moment_inertia / p.volume * m_i + m_i * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    for root in (S / 'assets', E / 'stage2/assets' / n):
        cd = root / f'{n}_convex'
        if not (root / f'{n}_convex_coacd').exists(): shutil.move(str(cd), str(root / f'{n}_convex_coacd'))
        else: shutil.rmtree(cd)
        cd.mkdir()
        for k, p in enumerate(parts): p.export(cd / f'part_{k:03d}.obj')
    tr = ET.parse(S / f'assets/{n}.urdf'); link = tr.getroot().find('link'); inert = link.find('inertial'); inert.find('origin').set('xyz', ' '.join(repr(float(x)) for x in com))
    for kk, ax in zip(('ixx', 'ixy', 'ixz', 'iyy', 'iyz', 'izz'), ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))): inert.find('inertia').set(kk, repr(float(I[ax])))
    for cc in list(link.findall('collision')): link.remove(cc)
    for k in range(len(parts)):
        cc = ET.SubElement(link, 'collision'); g = ET.SubElement(cc, 'geometry'); ET.SubElement(g, 'mesh', filename=f'{n}_convex/part_{k:03d}.obj')
    ET.indent(tr); tr.write(S / f'assets/{n}.urdf', encoding='utf-8', xml_declaration=True); shutil.copy2(S / f'assets/{n}.urdf', EP10 / f'{n}.urdf')
    rep[n] = dict(parts=len(parts), bands=NB, sectors=NA, base_thickness_cm=float((base_top - h0) * 100), height_cm=float((h1 - h0) * 100),
                  inner_radius_cm=np.round(R_in * 100, 2).tolist(), outer_radius_cm=np.round(R_out * 100, 2).tolist(), band_heights_cm=np.round((edges - h0) * 100, 2).tolist())
    print(n, json.dumps({k: v for k, v in rep[n].items()}), flush=True)
m = json.loads((E / 'input_manifest.json').read_text())
for o in m['objects']:
    nm = o['id']; o['source_urdf_sha256'] = sha(S / f'assets/{nm}.urdf')
    o['meshes'] = [x for x in o['meshes'] if x['kind'] == 'visual'] + [dict(source=str(p), sha256=sha(p), kind='collision') for p in sorted((S / f'assets/{nm}_convex').glob('part_*.obj'))]
m['scan_cup_replacement']['thin_wall_collision'] = rep; (E / 'input_manifest.json').write_text(json.dumps(m, indent=2))
json.dump(rep, open(EP10 / 'thin_wall_collision.json', 'w'), indent=1)
