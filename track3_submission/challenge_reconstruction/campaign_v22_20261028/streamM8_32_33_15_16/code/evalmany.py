"""Stream M8: score many existing actors (own scene, closed-loop rollout == evaluate_policy_component) vs target_final_v5.
usage: evalmany.py EP MESHDIR OUT.jsonl ACTOR_DIR..."""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import sys, json
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
def one(args):
    ep, md, ad = args
    try:
        import mres; env = mres.Env(dict(ep=ep, actor=ad, meshes=md)); r = env.evaluate(env.table0)
        return dict(actor=ad, cost=r['cost'], objects=r.get('objects'), err=r.get('error'))
    except Exception as ex: return dict(actor=ad, error=repr(ex)[:300])
if __name__ == '__main__':
    ep, md, out = int(sys.argv[1]), sys.argv[2], sys.argv[3]; ads = sys.argv[4:]
    with ProcessPoolExecutor(max_workers=min(30, len(ads)), mp_context=mp.get_context('spawn')) as pool, open(out, 'a') as f:
        for r in pool.map(one, [(ep, md, a) for a in ads]):
            f.write(json.dumps(r) + '\n'); f.flush(); print(json.dumps(r), flush=True)
