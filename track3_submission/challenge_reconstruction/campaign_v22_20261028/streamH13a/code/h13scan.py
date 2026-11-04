"""H13a step 1: full-horizon closed-loop rollouts of the v27 actor with a global time-advance of its phase reference, scored vs target_final_v12b_g15
(h13lib.per_frame == score_any objects; per-frame kit MPPE mean). usage: h13scan.py EP ACTOR MESHDIR OUT.json 'spec|spec|...'"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import sys, json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import h13res as R


def one(args):
    ep, actor, meshes, spec = args
    env = R.Env(dict(ep=ep, actor=actor, meshes=meshes, advance=spec))
    Q = env.rollout(env.table0)
    if Q is None: return dict(spec=spec, unstable=True)
    e = R.L.per_frame(ep, env.poses(Q), env.reg, env.tgt, want_mppe=True)
    o = {n: [round(float(e[n][0].mean()), 4), round(float(e[n][1].mean()), 3)] for n in env.names}
    return dict(spec=spec, objects=o, mppe=round(float(e['_mppe'].mean()), 4), max_cm=max(v[0] for v in o.values()), max_deg=max(v[1] for v in o.values()), extra=env.extra)


if __name__ == '__main__':
    ep, actor, meshes, out, specs = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5].split('|')
    with ProcessPoolExecutor(max_workers=len(specs), mp_context=mp.get_context('spawn')) as pool:
        res = list(pool.map(one, [(ep, actor, meshes, s) for s in specs]))
    for r in res: print('ADV', json.dumps(r), flush=True)
    Path(out).write_text(json.dumps(res, indent=1))
