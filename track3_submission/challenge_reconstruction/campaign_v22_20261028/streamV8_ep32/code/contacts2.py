"""V8: right-hand links touching the spoon per 0.5 s (actor's recorded rollout) + spoon-cup contact + target spoon/cup penetration.
usage: contacts2.py ACTOR_DIR"""
import sys, json, numpy as np, mujoco
from collections import Counter
from pathlib import Path
ad = Path(sys.argv[1]); r = json.loads((ad / 'evaluation/result.json').read_text()); m = mujoco.MjModel.from_xml_path(r['execution']['scene']); d = mujoco.MjData(m)
q = np.load(ad / 'evaluation/execution/trajectory.npz')['qpos']; sp = m.body('wooden_spoon').id; cu = m.body('blue_cup').id
for s0 in range(0, len(q), 10):
    c = Counter(); cup = 0; dmin = 0
    for k in range(s0, min(s0 + 10, len(q))):
        d.qpos[:] = q[k]; mujoco.mj_forward(m, d)
        for i in range(d.ncon):
            b1, b2 = m.geom_bodyid[d.contact[i].geom1], m.geom_bodyid[d.contact[i].geom2]
            if sp in (b1, b2):
                o = b2 if b1 == sp else b1; n = m.body(o).name
                if o == cu: cup += 1; dmin = min(dmin, d.contact[i].dist)
                elif n.startswith('robot_right'): c[n.replace('robot_right_right_', '')] += 1
                elif n.startswith('robot_left'): c['LEFT:' + n] += 1
    print(f'{s0/20:5.1f}s cup={cup}({dmin*1000:.1f}mm) ' + ' '.join(f'{k}:{v}' for k, v in sorted(c.items())))
