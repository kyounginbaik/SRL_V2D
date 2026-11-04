"""H13b: global TIME-ADVANCE of a shipped actor, scored vs target_final_v12b_g15 (mlibH.per_frame == score_any).
The actor is phase-RBF + state feedback: u(t) = phi(t/dur; centers) @ W + fb(state). Advancing every phase command by s frames is exact by
shifting the RBF centers: centers' = centers - s/(fps*dur)  =>  phi'(t) = phi(t + s/fps). Time-keyed wrapper fields (release_start_s,
multi-release starts) are shifted by -s/fps as well. Feedback weights, scene, start state, wrappers otherwise unchanged.
usage: tadv.py --ep E --actor A --meshes M --out O --shifts 0,0.5,... [--save S]  (--save: bake the best/given shift via mresH.save_actor)"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import sys, json, argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mresH as mres

TIME_KEYS = ('release_start_s',)


def extra_for(actor, ckpt, s):
    with np.load(ckpt or str(Path(actor) / 'training/policy.npz'), allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    dt = s / float(d['fps']); ex = {'centers': (d['centers'] - dt / float(d['duration_s'])).tolist()}
    for k in d:
        if k in TIME_KEYS or (k.startswith('multi_release') and k.endswith('start_s')):
            ex[k] = (np.asarray(d[k], float) - dt).tolist()
    return ex


def one(args):
    cfg, s = args
    cfg = dict(cfg, extra=extra_for(cfg['actor'], cfg.get('ckpt'), s)); env = mres.Env(cfg)
    r = env.evaluate(env.table0, keep=True)
    if 'err' not in r: return dict(shift=s, fail=r)
    e = mres.L.per_frame(env.ep, env.poses(r['Q']), env.reg, env.tgt, want_mppe=True)
    return dict(shift=s, objects={n: [round(v, 4) for v in r['objects'][n]] for n in env.names}, mppe=float(e['_mppe'].mean()), cost=r['cost'])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--ep', type=int, required=True); ap.add_argument('--actor', required=True); ap.add_argument('--meshes', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--shifts', default='0,0.5,0.75,0.9,1.0,1.25,1.5,2.0'); ap.add_argument('--save', type=float); ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = dict(ep=a.ep, actor=str(Path(a.actor).resolve()), meshes=a.meshes)
    if a.save is not None:
        ex = extra_for(cfg['actor'], None, a.save); env = mres.Env(dict(cfg, extra=ex))
        res = mres.save_actor(env, env.table0.copy(), out / 'actor', dict(mode='H13b time-advance', shift_frames=a.save, shifted_keys=sorted(ex)))
        (out / 'save.json').write_text(json.dumps(dict(shift=a.save, objects=res))); return
    shifts = [float(v) for v in a.shifts.split(',')]
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context('spawn')) as pool:
        rows = list(pool.map(one, [(cfg, s) for s in shifts]))
    for r in rows: print('SHIFT', json.dumps(r), flush=True)
    (out / 'tadv.json').write_text(json.dumps(rows, indent=1))


if __name__ == '__main__': main()
