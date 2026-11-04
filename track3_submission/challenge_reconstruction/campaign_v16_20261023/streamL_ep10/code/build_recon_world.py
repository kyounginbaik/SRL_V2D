"""Stream L (episode 10): recon-cup world with a thin-wall collision built from OUR reconstruction meshes.

Inputs (nothing under /data1/SRL_V2D/gt_best_per_object is read):
  EPD   = refinement_20261018/episode_000010/calibrated_inputs/episodes/episode_000010  (pre-scan recon-cup episode dir,
          visual meshes 2c04.. / 57bb.. = final_recon/episode_000010/assets/{beige,blue}_cup_visual.obj, 32 CoACD parts)
  FINAL = /data1/SRL_V2D/challenge_reconstruction/final_recon/episode_000010/assets (our submitted reconstruction cups)
Output: WORLD = streamL_ep10/world/episode_000010 (copy of EPD) with stage2/assets/<cup>/<cup>_convex replaced by thin-wall
sectors (NA sectors x NB bands + base disk) fitted to the radial profile of OUR visual mesh (CoACD parts kept as
<cup>_convex_coacd), streamL_ep10/world/source_assets/<cup>.urdf (+ parts) with inertia recomputed from the parts, and an
updated input_manifest.json (collision entries, source_urdf). metric_geometry.json, object poses, visual meshes unchanged.
Why: the 32-part CoACD hulls of the 0.7 mm shells bulge into the cavity (chord sagitta), so the beige cup bottoms out at
8.3 cm insertion while the reference nests 11 cm; the previous fix used the GT scan cups, which the user forbids.
usage: build_recon_world.py OUT_DIR [NA NB]"""
import json, shutil, hashlib, sys, numpy as np, trimesh
import xml.etree.ElementTree as ET
from pathlib import Path
R0 = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017')
EPD = R0 / 'refinement_20261018/episode_000010/calibrated_inputs/episodes/episode_000010'
FINAL = Path('/data1/SRL_V2D/challenge_reconstruction/final_recon/episode_000010/assets')
EXPECTED = dict(beige_cup='2c041e69fbd3be79fa953609276ffc4a54a513ec2ca7e7f31020a78ab6871431',
                blue_cup='57bb269a06a44f9f56ea59c9eea8b0d704b20d5a8cfd8bbc89f73f5e2f2b1d8f')
OUT = Path(sys.argv[1]).resolve(); NA = int(sys.argv[2]) if len(sys.argv) > 2 else 24; NB = int(sys.argv[3]) if len(sys.argv) > 3 else 10
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
W = OUT / 'episode_000010'; SA = OUT / 'source_assets'
if W.exists(): raise SystemExit('refusing to overwrite ' + str(W))
shutil.copytree(EPD, W); SA.mkdir(parents=True)
rep = {}
for n in ('blue_cup', 'beige_cup'):
    vis = FINAL / f'{n}_visual.obj'; assert sha(vis) == EXPECTED[n], n
    assert sha(W / 'stage2/assets' / n / f'{n}_visual.obj') == EXPECTED[n], 'episode-dir visual is not our reconstruction'
    m = trimesh.load(vis, force='mesh', process=False)
    P = trimesh.sample.sample_surface(m, 300000, seed=0)[0]; c = P.mean(0); w, Ue = np.linalg.eigh(np.cov((P - c).T)); a = Ue[:, -1]
    h = (P - c) @ a; rr = np.linalg.norm((P - c) - np.outer(h, a), axis=1)
    if rr[h > np.percentile(h, 85)].mean() < rr[h < np.percentile(h, 15)].mean(): a = -a; h = -h
    e1 = np.cross(a, [1, 0, 0]); e1 = e1 / np.linalg.norm(e1) if np.linalg.norm(e1) > .1 else np.cross(a, [0, 1, 0]) / np.linalg.norm(np.cross(a, [0, 1, 0])); e2 = np.cross(a, e1)
    ang = np.arctan2((P - c) @ e2, (P - c) @ e1); h0, h1 = h.min(), h.max()
    def radii(lo, hi):
        s = (h >= lo) & (h < hi); ins, outs = [], []
        for k in range(36):
            q = s & (ang >= -np.pi + k * 2 * np.pi / 36) & (ang < -np.pi + (k + 1) * 2 * np.pi / 36)
            if q.sum() > 5: ins.append(rr[q].min()); outs.append(rr[q].max())
        return (np.median(ins), np.median(outs)) if ins else (np.nan, np.nan)
    # floor: highest near-axis surface (rr < 1.5 cm) within the lowest 3 cm; the base disk spans h0 .. floor (+1 mm), >= 3 mm
    near = (rr < 0.015) & (h < h0 + 0.03); base_top = max(float(h[near].max()) + 0.001, h0 + 0.003) if near.any() else h0 + 0.004
    edges = np.linspace(base_top, h1, NB + 1); R_in, R_out = [], []
    for z in edges:   # wall radii from points above the floor only (the floor would fake a tiny inner radius)
        ri, ro = radii(max(z - 0.004, base_top), z + 0.004); R_in.append(ri); R_out.append(ro)
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
    rb = radii(h0, base_top)[1]; rb = rb if np.isfinite(rb) else R_out[0]; base = []
    for th in np.linspace(0, 2 * np.pi, 33)[:-1]:
        d = np.cos(th) * e1 + np.sin(th) * e2; base += [c + a * h0 + d * rb, c + a * base_top + d * R_out[0]]
    parts.append(trimesh.convex.convex_hull(np.array(base)))
    vol = np.array([p.volume for p in parts]); rho = 0.1 / vol.sum(); com = sum(p.volume * p.center_mass for p in parts) / vol.sum(); I = np.zeros((3, 3))
    for p in parts:
        m_i = rho * p.volume; dd = p.center_mass - com; I += p.moment_inertia / p.volume * m_i + m_i * (np.dot(dd, dd) * np.eye(3) - np.outer(dd, dd))
    for root in (W / 'stage2/assets' / n, SA):
        cd = root / f'{n}_convex'
        if cd.exists(): shutil.move(str(cd), str(root / f'{n}_convex_coacd'))
        cd.mkdir()
        for k, p in enumerate(parts): p.export(cd / f'part_{k:03d}.obj')
    shutil.copy2(vis, SA / f'{n}_visual.obj')
    tr = ET.parse(FINAL / f'{n}.urdf'); link = tr.getroot().find('link'); inert = link.find('inertial'); inert.find('origin').set('xyz', ' '.join(repr(float(x)) for x in com))
    for kk, ax in zip(('ixx', 'ixy', 'ixz', 'iyy', 'iyz', 'izz'), ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))): inert.find('inertia').set(kk, repr(float(I[ax])))
    for cc in list(link.findall('collision')): link.remove(cc)
    for k in range(len(parts)):
        cc = ET.SubElement(link, 'collision'); g = ET.SubElement(cc, 'geometry'); ET.SubElement(g, 'mesh', filename=f'{n}_convex/part_{k:03d}.obj')
    ET.indent(tr); tr.write(SA / f'{n}.urdf', encoding='utf-8', xml_declaration=True)
    rep[n] = dict(visual_source=str(vis), visual_sha256=sha(vis), parts=len(parts), bands=NB, sectors=NA, mass_kg=0.1, com_body=com.tolist(),
                  base_thickness_cm=float((base_top - h0) * 100), height_cm=float((h1 - h0) * 100), axis_body=a.tolist(),
                  inner_radius_cm=np.round(R_in * 100, 2).tolist(), outer_radius_cm=np.round(R_out * 100, 2).tolist(), band_heights_cm=np.round((edges - h0) * 100, 2).tolist(),
                  urdf=str(SA / f'{n}.urdf'), urdf_sha256=sha(SA / f'{n}.urdf'), parts_sha256_first=sha(SA / f'{n}_convex/part_000.obj'))
    print(n, json.dumps({k: v for k, v in rep[n].items() if k not in ('axis_body', 'com_body')}), flush=True)
man = json.loads((W / 'input_manifest.json').read_text())
for o in man['objects']:
    nm = o['id']; o['source_urdf'] = str(SA / f'{nm}.urdf'); o['source_urdf_sha256'] = sha(SA / f'{nm}.urdf')
    vis_entries = [x for x in o['meshes'] if x['kind'] == 'visual']; assert len(vis_entries) == 1 and vis_entries[0]['sha256'] == EXPECTED[nm]
    o['meshes'] = vis_entries + [dict(source=str(p), sha256=sha(p), kind='collision') for p in sorted((SA / f'{nm}_convex').glob('part_*.obj'))]
man['setup'] = str(W / 'setup'); man['scene_recon'] = str(W / 'stage2/scene_recon.json')
man['recon_cup_thin_wall_collision'] = dict(note='blue_cup and beige_cup collision replaced by thin-wall convex sectors fitted to OUR reconstruction visual meshes (final_recon/episode_000010/assets); no scan geometry; visual meshes and object poses unchanged', derived_from=str(EPD), report=rep)
(W / 'input_manifest.json').write_text(json.dumps(man, indent=2))
json.dump(dict(report=rep, episode_dir=str(W), source_assets=str(SA), derived_from=str(EPD), NA=NA, NB=NB,
               object_poses_unchanged=True, metric_geometry_sha256=sha(W / 'metric_geometry.json'), epd_metric_geometry_sha256=sha(EPD / 'metric_geometry.json')),
          open(OUT / 'recon_cup_thin_wall.json', 'w'), indent=1)
print('done', W)
