"""which robot bodies (by side) touch the spoon / pot per 0.25 s, replaying an actor's evaluated qpos (mj_forward per frame)."""
import sys, json, numpy as np, mujoco
from pathlib import Path
a = Path(sys.argv[1]); r = json.loads((a / 'evaluation/result.json').read_text())
m = mujoco.MjModel.from_xml_path(r['execution']['scene']); d = mujoco.MjData(m)
q = np.load(a / 'evaluation/execution/trajectory.npz')['qpos']
def root(b):
    return m.body(b).name
sp = {i for i in range(m.nbody) if 'wooden_spoon' in m.body(i).name}
pot = {i for i in range(m.nbody) if 'white_pot' in m.body(i).name}
for k0 in range(0, len(q), 5):
    hits = {}
    for k in range(k0, min(k0 + 5, len(q))):
        d.qpos[:] = q[k]; mujoco.mj_forward(m, d)
        for c in d.contact[:d.ncon]:
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            for x, y in ((b1, b2), (b2, b1)):
                if x in sp and m.body(y).name.startswith('robot_left'):
                    hits['L-spoon:' + m.body(y).name.replace('robot_left_left_', '')] = 1
                if x in sp and y in pot: hits['spoon-pot'] = 1
    print(f'{k0/20:5.2f}', ' '.join(sorted(hits)))
