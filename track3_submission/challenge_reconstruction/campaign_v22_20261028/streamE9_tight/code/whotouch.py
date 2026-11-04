"""E9: per 0.5 s window, number of contact pairs between each object and the left / right hand / other bodies in an actor's recorded rollout.
usage: whotouch.py ACTOR_DIR OBJ1 [OBJ2 ...]"""
import sys, json, numpy as np, mujoco
from pathlib import Path
ad = Path(sys.argv[1]); objs = sys.argv[2:]
r = json.loads((ad / 'evaluation/result.json').read_text()); m = mujoco.MjModel.from_xml_path(r['execution']['scene']); d = mujoco.MjData(m)
q = np.load(ad / 'evaluation/execution/trajectory.npz')['qpos']; ob = {o: m.body(o).id for o in objs}
def side(b):
    n = m.body(b).name
    return 'L' if n.startswith('robot_left') else 'R' if n.startswith('robot_right') else n
for s0 in range(0, len(q), 10):
    c = {o: {} for o in objs}
    for k in range(s0, min(s0 + 10, len(q))):
        d.qpos[:] = q[k]; mujoco.mj_forward(m, d)
        for i in range(d.ncon):
            b1, b2 = m.geom_bodyid[d.contact[i].geom1], m.geom_bodyid[d.contact[i].geom2]
            for o, bo in ob.items():
                if bo in (b1, b2):
                    s = side(b2 if b1 == bo else b1); c[o][s] = c[o].get(s, 0) + 1
    print(f'{s0/20:5.1f}s', '  '.join(f'{o}:{c[o]}' for o in objs))
