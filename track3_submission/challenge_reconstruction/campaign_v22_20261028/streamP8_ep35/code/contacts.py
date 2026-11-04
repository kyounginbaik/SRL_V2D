"""P8: per-frame hand-spoon contacts of an evaluated rollout (execution dir). usage: contacts.py EXEC_DIR [step]"""
import sys, numpy as np, mujoco, collections
from pathlib import Path
E = Path(sys.argv[1]); step = int(sys.argv[2]) if len(sys.argv) > 2 else 20
m = mujoco.MjModel.from_xml_path(str(E / 'scene_physics.xml')); d = mujoco.MjData(m)
z = np.load(E / 'trajectory.npz')
def bname(g): return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ''
for f in range(0, len(z['qpos']), step):
    d.qpos[:] = z['qpos'][f]; mujoco.mj_forward(m, d); s = collections.Counter()
    for i in range(d.ncon):
        c = d.contact[i]; a, b = bname(c.geom1), bname(c.geom2)
        if 'spoon' in a or 'spoon' in b:
            o = b if 'spoon' in a else a; s[o.replace('robot_', '')] += 1
    print(f'f{f:3d} t{f/20:5.2f}', dict(s))
