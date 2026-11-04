"""Spoon-pot (and object-table) penetration of a reference trajectory in a scene: usage pen.py SCENE REF_TRAJ_NPZ [STEP]"""
import sys, numpy as np, mujoco
m = mujoco.MjModel.from_xml_path(sys.argv[1]); d = mujoco.MjData(m); z = np.load(sys.argv[2]); st = int(sys.argv[3]) if len(sys.argv) > 3 else 10
q = z['qpos']; out = []
for k in range(0, len(q), st):
    d.qpos[:] = q[k]; mujoco.mj_forward(m, d); pairs = {}
    for c in d.contact[:d.ncon]:
        b1, b2 = m.body(int(m.geom_bodyid[c.geom1])).name, m.body(int(m.geom_bodyid[c.geom2])).name
        if b1.startswith('robot') or b2.startswith('robot'): continue
        key = tuple(sorted((b1, b2))); pairs[key] = min(pairs.get(key, 0), c.dist)
    print(k, ' '.join(f'{a}-{b}:{v*1000:.0f}mm' for (a, b), v in pairs.items()))
