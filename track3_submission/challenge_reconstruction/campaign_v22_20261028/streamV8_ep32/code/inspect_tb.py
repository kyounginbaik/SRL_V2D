"""V8: roll out the actor with residual tables (mres layout) and print per 0.5 s spoon error, spoon height, right-hand links touching the spoon, cup contact.
usage: inspect_tb.py ACTOR MESHES TABLE.npz [TABLE2.npz ...]"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import sys, numpy as np, mujoco
from collections import Counter
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamM8_32_33_15_16/code')
import mres
env = mres.Env(dict(ep=32, actor=sys.argv[1], meshes=sys.argv[2])); m = env.m; d = mujoco.MjData(m); sp = m.body('wooden_spoon').id; cu = m.body('blue_cup').id
for tp in sys.argv[3:]:
    tb = np.zeros((env.T, mres.NCH)) if tp == 'zero' else np.load(tp)['table']; Q = env.rollout(tb); c, res, e, A = env.score(Q); print('==', tp, res)
    for s0 in range(0, env.T, 10):
        cc = Counter(); cup = 0
        for k in range(s0, min(s0 + 10, env.T)):
            d.qpos[:] = Q[k]; mujoco.mj_forward(m, d)
            for i in range(d.ncon):
                b1, b2 = m.geom_bodyid[d.contact[i].geom1], m.geom_bodyid[d.contact[i].geom2]
                if sp in (b1, b2):
                    o = b2 if b1 == sp else b1; n = m.body(o).name
                    if o == cu: cup += 1
                    elif n.startswith('robot_right'): cc[n.replace('robot_right_right_', '')] += 1
                    else: cc[n[:12]] += 1
        print(f"{s0/20:5.1f} {e['wooden_spoon'][0][s0:s0+10].mean():6.2f}/{e['wooden_spoon'][1][s0:s0+10].mean():5.1f} z={Q[s0, env.oa['wooden_spoon']+2]*100:5.1f} cup={cup:2d} " + ' '.join(f'{k}:{v}' for k, v in sorted(cc.items())), flush=True)
