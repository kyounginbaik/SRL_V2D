"""Stream M8: which bodies touch the spoon per frame in an actor's recorded rollout (qpos replay + mj_forward).
usage: contacts.py ACTOR_DIR [OBJ]"""
import sys, json, numpy as np, mujoco
from collections import Counter
from pathlib import Path
ad = Path(sys.argv[1]); obj = sys.argv[2] if len(sys.argv) > 2 else 'wooden_spoon'
r = json.loads((ad / 'evaluation/result.json').read_text()); m = mujoco.MjModel.from_xml_path(r['execution']['scene']); d = mujoco.MjData(m)
q = np.load(ad / 'evaluation/execution/trajectory.npz')['qpos']; ob = m.body(obj).id
def root(b):
    n = m.body(b).name
    for p in ('robot_left', 'robot_right'):
        if n.startswith(p): return p + ':' + n.split('_', 3)[-1]
    return n
for s0 in range(0, len(q), 10):
    c = Counter()
    for k in range(s0, min(s0 + 10, len(q))):
        d.qpos[:] = q[k]; mujoco.mj_forward(m, d)
        for i in range(d.ncon):
            g1, g2 = d.contact[i].geom1, d.contact[i].geom2; b1, b2 = m.geom_bodyid[g1], m.geom_bodyid[g2]
            if ob in (b1, b2):
                o = b2 if b1 == ob else b1; c[root(o)] += 1
    left = {k: v for k, v in c.items() if k.startswith('robot_left')}
    oth = {k: v for k, v in c.items() if not k.startswith('robot_')}
    print(f'{s0/20:5.1f}s right={sum(v for k,v in c.items() if k.startswith("robot_right"))} LEFT={left} other={oth}')
