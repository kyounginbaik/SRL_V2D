"""F9 helper: which hand touches which object (contact frames per object x side) in a base_eval rollout, plus per-1s window errors.
usage: handsF9.py EP ACTOR MESHES EVAL_NPZ"""
import sys, json
from pathlib import Path
import numpy as np, mujoco
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mresF9 as mres
ep, actor, meshes, evf = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
env = mres.Env(dict(ep=ep, actor=str(Path(actor).resolve()), meshes=meshes))
Q = np.load(evf)['Q']; m = env.m; d = mujoco.MjData(m)
body_side = {}
for b in range(m.nbody):
    nm = m.body(b).name
    for s in ('right', 'left'):
        if nm.startswith(f'robot_{s}_'): body_side[b] = s
obj_body = {n: int(m.joint(f'{n}_free').bodyid[0]) for n in env.names}
cnt = {n: {'right': np.zeros(len(Q), bool), 'left': np.zeros(len(Q), bool)} for n in env.names}
for k in range(len(Q)):
    d.qpos[:] = Q[k]; mujoco.mj_forward(m, d)
    for i in range(d.ncon):
        c = d.contact[i]; b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
        for n, ob in obj_body.items():
            for x, y in ((b1, b2), (b2, b1)):
                if x == ob and y in body_side: cnt[n][body_side[y]][k] = True
out = {}
for n in env.names:
    out[n] = {}
    for s in ('right', 'left'):
        f = np.flatnonzero(cnt[n][s])
        out[n][s] = dict(frames=int(len(f)), first_s=float(f[0] / env.fps) if len(f) else None, last_s=float(f[-1] / env.fps) if len(f) else None)
print(json.dumps(out, indent=1))
