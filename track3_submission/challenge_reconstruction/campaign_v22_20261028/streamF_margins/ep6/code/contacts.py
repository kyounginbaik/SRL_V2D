"""Per-frame hand-object contact census of an executed trajectory (kinematic replay + mj_forward collision pass).
usage: contacts.py TEACHER_DIR [STEP] [--npz OUT]  -> prints per STEP frames: object heights and contacts L/R x pot/spoon; --npz saves the per-frame table"""
import os; os.environ['OPENBLAS_NUM_THREADS'] = '1'
import sys, json, numpy as np, mujoco
from pathlib import Path
t = Path(sys.argv[1]); step = int(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else 5
r = json.loads((t / 'result.json').read_text()); tr = np.load(r['execution']['trajectory']); m = mujoco.MjModel.from_xml_path(r['execution']['scene']); d = mujoco.MjData(m)
names = [str(x) for x in tr['object_names']]; n = len(tr['qpos']); C = np.zeros((n, 2, len(names)), int)
def grp(b):
    if b.startswith('robot_left'): return 0
    if b.startswith('robot_right'): return 1
    return None
for k in range(n):
    d.qpos[:] = tr['qpos'][k]; mujoco.mj_forward(m, d)
    for c in d.contact[:d.ncon]:
        b1, b2 = m.body(int(m.geom_bodyid[c.geom1])).name, m.body(int(m.geom_bodyid[c.geom2])).name
        for a, b in ((b1, b2), (b2, b1)):
            h = grp(a)
            if h is not None and b in names and c.dist < 0.002: C[k, h, names.index(b)] += 1
for k in range(0, n, step):
    print(f'{k:3d} {k/20:5.2f}s z ' + ' '.join(f'{o[:5]} {tr["object_positions"][k, i, 2]*100:5.1f}' for i, o in enumerate(names)) + ' | L ' + ' '.join(f'{C[k,0,i]:2d}' for i in range(len(names))) + ' R ' + ' '.join(f'{C[k,1,i]:2d}' for i in range(len(names))))
if '--npz' in sys.argv: np.savez(sys.argv[sys.argv.index('--npz') + 1], contacts=C, names=names)
