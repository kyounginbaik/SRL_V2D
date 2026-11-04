"""Swap one board's COLLISION parts (visual mesh unchanged) in a frozen scene_physics.xml with the slot boxes; re-creates every
explicit <pair> per new part; recomputes the inertial from the boxes (mass unchanged).
usage: swap_board_collision.py IN_SCENE OUT_SCENE OBJECT PARTS_DIR [OBJECT PARTS_DIR ...]"""
import sys, copy, re, os
import numpy as np, trimesh
from pathlib import Path
import xml.etree.ElementTree as ET
inp, out = Path(sys.argv[1]), Path(sys.argv[2]); jobs = list(zip(sys.argv[3::2], sys.argv[4::2]))
tree = ET.parse(inp); root = tree.getroot(); asset = root.find('asset')
for P, D in jobs:
    parts = sorted(Path(D).glob('part_*.obj'))
    ms = [m for m in asset.findall('mesh') if re.match(rf'^{P}_collision_\d+$', m.get('name', ''))]; tmpl = ms[0]; pos = list(asset).index(ms[0])
    for m in ms: asset.remove(m)
    for k, p in enumerate(parts):
        m = copy.deepcopy(tmpl); m.set('name', f'{P}_collision_{k}'); m.set('file', str(p)); asset.insert(pos + k, m)
    body = next(b for b in root.iter('body') if b.get('name') == P)
    gs = [g for g in body.findall('geom') if re.match(rf'^{P}_collision_\d+$', g.get('name', ''))]; gt = gs[0]; gpos = list(body).index(gs[0])
    for g in gs: body.remove(g)
    for k in range(len(parts)):
        g = copy.deepcopy(gt); g.set('name', f'{P}_collision_{k}'); g.set('mesh', f'{P}_collision_{k}'); body.insert(gpos + k, g)
    inert = body.find('inertial'); mass = float(inert.get('mass')); com = None
    if not os.environ.get('KEEP_INERTIA'):
        hulls = [trimesh.load(p, force='mesh') for p in parts]; vol = np.array([h.volume for h in hulls]); rho = mass / vol.sum()
        com = sum(h.volume * h.center_mass for h in hulls) / vol.sum(); I = np.zeros((3, 3))
        for h in hulls:
            mi = rho * h.volume; d = h.center_mass - com; I += h.moment_inertia * rho + mi * (d @ d * np.eye(3) - np.outer(d, d))
        inert.attrib.clear(); inert.set('pos', ' '.join(f'{x:.9g}' for x in com)); inert.set('mass', f'{mass:g}')
        inert.set('fullinertia', ' '.join(f'{x:.9g}' for x in (I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2])))
    rx = re.compile(rf'^{P}_collision_\d+$'); nold = nnew = 0
    for contact in root.findall('contact'):
        groups = {}; order = []
        for p in contact.findall('pair'):
            side = 'geom1' if rx.match(p.get('geom1', '')) else 'geom2' if rx.match(p.get('geom2', '')) else None
            if side is None: continue
            nold += 1; other = 'geom2' if side == 'geom1' else 'geom1'
            key = (side, p.get(other), tuple(sorted((k, v) for k, v in p.attrib.items() if k not in ('name', side))))
            if key not in groups: groups[key] = p; order.append(key)
            contact.remove(p)
        for key in order:
            side = key[0]; t = groups[key]; old = t.get(side)
            for k in range(len(parts)):
                q = copy.deepcopy(t); q.set(side, f'{P}_collision_{k}')
                if q.get('name'): q.set('name', t.get('name').replace(old, f'{P}_collision_{k}'))
                contact.append(q); nnew += 1
    print(P, 'parts', len(ms), '->', len(parts), 'pairs', nold, '->', nnew, 'mass', mass, 'com', None if com is None else np.round(com * 100, 2))
tree.write(out, encoding='utf-8')
