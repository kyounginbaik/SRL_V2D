"""X8: put the X8 collision parts into a frozen ep10 scene_physics.xml (text edit; everything else byte-identical):
collision mesh files of blue_cup / beige_cup -> WORLD/source_assets/<cup>_convex/part_NNN.obj (same count/names), and the cup
body <inertial> -> pos + fullinertia of WORLD/source_assets/<cup>.urdf (mass 0.1 unchanged).  Visual mesh paths untouched.
usage: swap_scene_x8.py OLD_SCENE WORLD OUT_SCENE"""
import sys, re
from pathlib import Path
import xml.etree.ElementTree as ET
old, W, out = Path(sys.argv[1]), Path(sys.argv[2]).resolve(), Path(sys.argv[3])
if out.exists(): raise SystemExit('exists ' + str(out))
s = old.read_text()
for n in ('blue_cup', 'beige_cup'):
    rx = re.compile(rf'file="[^"]*/{n}_convex/part_(\d{{3}})\.obj"')
    s, k = rx.subn(lambda m: f'file="{W}/source_assets/{n}_convex/part_{m.group(1)}.obj"', s); assert k == 241, (n, k)
    u = ET.parse(W / 'source_assets' / f'{n}.urdf').getroot().find('link/inertial'); I = u.find('inertia')
    assert u.find('mass').get('value') == '0.1'
    new = (f'<inertial pos="{u.find("origin").get("xyz")}" mass="0.1" fullinertia="' + ' '.join(I.get(k) for k in ('ixx', 'iyy', 'izz', 'ixy', 'ixz', 'iyz')) + '"/>')
    rb = re.compile(rf'(<body name="{n}"[^>]*>\s*)<inertial [^>]*/>')
    s, k = rb.subn(lambda m: m.group(1) + new, s); assert k == 1, (n, k)
assert 'gt_best_per_object' not in s
out.write_text(s); print('wrote', out)
