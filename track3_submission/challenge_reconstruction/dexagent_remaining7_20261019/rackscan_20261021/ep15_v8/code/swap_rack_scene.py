"""Swap the plastic_dish_rack geometry inside a frozen teacher scene_physics.xml: mesh assets, body inertial and geoms, and
every explicit contact <pair> that used an old rack collision part (re-created per new part with the partner's attributes).
Body pose, freejoint, keyframe and everything else are kept byte-for-byte in meaning.
usage: swap_rack_scene.py OLD_SCENE MESH_DIR(plastic_dish_rack_visual.obj + plastic_dish_rack_convex_repaired/) URDF OUT_SCENE"""
import sys, re, copy
from pathlib import Path
import xml.etree.ElementTree as ET
old, A, URDF, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
tree = ET.parse(old); root = tree.getroot(); P = 'plastic_dish_rack'
parts = sorted((A / f'{P}_convex_repaired').glob('part_*.obj'))
asset = root.find('asset'); ms = [m for m in asset.findall('mesh') if m.get('name', '').startswith(P + '_')]
vis_t = next(m for m in ms if m.get('name') == f'{P}_visual_0'); col_t = next(m for m in ms if m.get('name') == f'{P}_collision_0')
pos = list(asset).index(ms[0])
for m in ms: asset.remove(m)
new = [copy.deepcopy(vis_t)]; new[0].set('file', str(A / f'{P}_visual.obj'))
for k, p in enumerate(parts):
    m = copy.deepcopy(col_t); m.set('name', f'{P}_collision_{k}'); m.set('file', str(p)); new.append(m)
for i, m in enumerate(new): asset.insert(pos + i, m)
body = next(b for b in root.iter('body') if b.get('name') == P)
gs = [g for g in body.findall('geom') if g.get('name', '').startswith(f'{P}_collision_')]; gt = copy.deepcopy(gs[0]); gpos = list(body).index(gs[0])
for g in gs: body.remove(g)
for k in range(len(parts)):
    g = copy.deepcopy(gt); g.set('name', f'{P}_collision_{k}'); g.set('mesh', f'{P}_collision_{k}'); body.insert(gpos + k, g)
u = ET.parse(URDF).getroot().find('link/inertial'); I = {k: u.find('inertia').get(k) for k in ('ixx', 'iyy', 'izz', 'ixy', 'ixz', 'iyz')}
inert = body.find('inertial'); inert.attrib.clear(); inert.set('pos', u.find('origin').get('xyz')); inert.set('mass', u.find('mass').get('value'))
inert.set('fullinertia', ' '.join(I[k] for k in ('ixx', 'iyy', 'izz', 'ixy', 'ixz', 'iyz')))
rx = re.compile(rf'^{P}_collision_\d+$'); n_old_pairs = n_new_pairs = 0
for contact in root.findall('contact'):
    pairs = contact.findall('pair'); groups = {}; order = []
    for p in pairs:
        side = 'geom1' if rx.match(p.get('geom1', '')) else 'geom2' if rx.match(p.get('geom2', '')) else None
        if side is None: continue
        n_old_pairs += 1; other = 'geom2' if side == 'geom1' else 'geom1'
        key = (side, p.get(other), tuple(sorted((k, v) for k, v in p.attrib.items() if k not in ('name', side))))
        if key not in groups: groups[key] = p; order.append(key)
        contact.remove(p)
    for key in order:
        side, partner, _ = key; tmpl = groups[key]; oldg = tmpl.get(side)
        for k in range(len(parts)):
            q = copy.deepcopy(tmpl); q.set(side, f'{P}_collision_{k}')
            if q.get('name'): q.set('name', tmpl.get('name').replace(oldg, f'{P}_collision_{k}'))
            contact.append(q); n_new_pairs += 1
tree.write(out, encoding='utf-8')
print(dict(old_meshes=len(ms), new_parts=len(parts), old_pairs=n_old_pairs, new_pairs=n_new_pairs))
