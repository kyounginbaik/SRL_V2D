"""per-second contact summary of a run: which hand side touches which object. usage: contacts_d.py RUN_DIR"""
import sys, json, numpy as np, mujoco
from pathlib import Path
r = json.loads((Path(sys.argv[1]) / 'result.json').read_text()); m = mujoco.MjModel.from_xml_path(r['execution']['scene']); tr = np.load(r['execution']['trajectory'])
d = mujoco.MjData(m); objs = [str(x) for x in tr['object_names']]; n = len(tr['qpos']); hit = {}
for k in range(n):
    d.qpos[:] = tr['qpos'][k]; mujoco.mj_forward(m, d)
    for c in d.contact[:d.ncon]:
        if c.dist > 0.002: continue
        b = [m.body(int(m.geom_bodyid[g])).name for g in (c.geom1, c.geom2)]
        for o in objs:
            for s in ('left', 'right'):
                if (b[0] == o and b[1].startswith('robot_' + s)) or (b[1] == o and b[0].startswith('robot_' + s)): hit.setdefault((s, o), set()).add(k)
            if o in b and any(x != o and x in objs for x in b): hit.setdefault(('objobj', o), set()).add(k)
for key, fr in sorted(hit.items()):
    fr = np.array(sorted(fr)); print(key, 'frames', fr.min(), fr.max(), 'n', len(fr), 'per-sec', [int(((fr >= s) & (fr < s + 20)).sum()) for s in range(0, n, 20)])
