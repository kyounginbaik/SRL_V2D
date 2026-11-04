"""P8: hand-spoon contacts (+ normal force sum) for a saved qpos array. usage: contactsQ.py SCENE Q.npy f0 f1 step"""
import sys, numpy as np, mujoco, collections
m = mujoco.MjModel.from_xml_path(sys.argv[1]); d = mujoco.MjData(m); Q = np.load(sys.argv[2]); f0, f1, st = map(int, sys.argv[3:6])
def bname(g): return (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or '').replace('robot_', '').replace('right_right_', 'R.').replace('left_left_', 'L.')
for f in range(f0, f1, st):
    d.qpos[:] = Q[f]; mujoco.mj_forward(m, d); s = collections.Counter()
    for i in range(d.ncon):
        c = d.contact[i]; a, b = bname(c.geom1), bname(c.geom2)
        if 'spoon' in a or 'spoon' in b: s[b if 'spoon' in a else a] += 1
    print(f'f{f:3d} t{f/20:5.2f}', dict(sorted(s.items())))
