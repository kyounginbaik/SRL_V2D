"""E9: penetration of the TARGET pose (target_final_v6 in our body frames, physics world via the scorer alignment of a rollout) for OBJ
at given frames, against all other geoms (rack, table, other object at its achieved pose). usage: restpen.py EP ACTOR MESHES TABLE OBJ f1,f2,..."""
import sys, json, numpy as np, mujoco
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mresE9 as mres
from scipy.spatial.transform import Rotation as R
ep, actor, meshes, table, obj, frames = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], [int(x) for x in sys.argv[6].split(',')]
env = mres.Env(dict(ep=ep, actor=actor, meshes=meshes, table=table if table != '-' else None))
r = env.evaluate(env.table0, keep=True); Q = r['Q']; A = r['A']; tp = mres.L.target_phys(ep, env.reg, A, env.tgt)[obj]
m = env.m; d = mujoco.MjData(m); a = env.oa[obj]; ob = m.body(obj).id
for k in frames:
    for lab, T in (('achieved', None), ('target', tp[k])):
        d.qpos[:] = Q[k]
        if T is not None: d.qpos[a:a + 3] = T[:3, 3]; d.qpos[a + 3:a + 7] = R.from_matrix(T[:3, :3]).as_quat(scalar_first=True)
        mujoco.mj_forward(m, d); pen = {}
        for i in range(d.ncon):
            c = d.contact[i]; b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if ob in (b1, b2):
                o = m.body(b2 if b1 == ob else b1).name; pen[o] = min(pen.get(o, 0.0), c.dist)
        dp = np.linalg.norm(tp[k][:3, 3] - Q[k, a:a + 3]) * 100
        print(k, lab, 'body-origin offset cm %.2f' % dp, {o: round(v * 1000, 1) for o, v in pen.items()}, '(min dist mm)')
