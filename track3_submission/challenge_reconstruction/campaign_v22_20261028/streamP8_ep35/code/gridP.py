"""Stream P8 (v23) ep35: parallel grid of phase-table edits on an actor (uses streamK8 ilcK.Env unchanged, imported).
Each candidate = sum of windowed constant offsets on table channels (0-2 wrist pos m, 3-5 wrist hinge rad (chart coords), 6 right non-thumb
flexion rad, 7 right thumb flexion rad), smoothstep ramps. Prints full / per-window scorer-identical errors (tlibK) and saves results.
usage: gridP.py --actor DIR --out DIR --spec JSON_FILE [--table npz] [--workers N]
spec file: list of candidates, each a list of terms {"ch":[6,7],"amp":0.1,"t0":2.0,"t1":9.5,"ramp":0.3}"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import sys, json, argparse, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK8_35_36_37/code')
import ilcK
sm = ilcK.sm
W = None
BASEQ = None
def init(cfg):
    global W, BASEQ; W = ilcK.Env(cfg)
    names = [str(x) for x in W.d['actuator_names']]
    W.la = [names.index(f'robot_left_left_{j}_position') for j in ('pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z')]
    W.extra = np.zeros((W.T, 6)); _w = W.weights
    def weights(table, _w=_w):
        Wt = _w(table); tv = np.stack([np.interp(W.tc, W.t, W.extra[:, c]) for c in range(6)], 1)
        for i, a in enumerate(W.la): Wt[:, a] += tv[:, i] / W.scale[a]
        return Wt
    W.weights = weights
    if cfg.get('need_base'): BASEQ = W.rollout(W.table0)
def roll_delta(env, Q, phi_deg, axis='x'):
    """wrist table delta (T x 6) that rotates the hand rigidly about the EXECUTED spoon's body axis by phi(t) (chart-aware, as ilcK)."""
    from scipy.spatial.transform import Rotation as R
    sc = env.sc; Ra = R.from_quat(Q[:, sc + 3:sc + 7], scalar_first=True); pa = Q[:, sc:sc + 3]; qw = Q[:, env.wq[3:]]; pw = Q[:, env.wq[:3]]
    out = np.zeros((env.T, 6))
    for k in np.flatnonzero(np.abs(phi_deg) > 1e-9):
        D = Ra[k] * R.from_euler(axis, phi_deg[k], degrees=True) * Ra[k].inv(); pn = pa[k] + D.apply(pw[k] - pa[k])
        Rn = env.Ac.inv() * D * env.Ac * R.from_euler('XYZ', qw[k]); eu = Rn.as_euler('XYZ'); eu = qw[k] + (eu - qw[k] + np.pi) % (2 * np.pi) - np.pi
        out[k, :3] = pn - pw[k]; out[k, 3:] = eu - qw[k]
    return out
def table_of(env, base, terms):
    t = base.copy(); env.extra = np.zeros((env.T, 6))
    for tm in terms:
        r = tm.get('ramp', 0.3); w = sm((env.t - tm['t0']) / r) * (1 - sm((env.t - (tm['t1'] - r)) / r)) if tm['t1'] < env.t[-1] else sm((env.t - tm['t0']) / r)
        if 'lch' in tm:
            for c in tm['lch']: env.extra[:, c] += tm['amp'] * w
            continue
        if 'roll' in tm: t[:, :6] += roll_delta(env, BASEQ, tm['roll'] * w, tm.get('axis', 'x')); continue
        for c in tm['ch']: t[:, c] += tm['amp'] * w
    return t
def ev(args):
    i, terms = args; tb = table_of(W, W.table0, terms)
    try:
        Q = W.rollout(tb)
        if Q is None: return dict(i=i, terms=terms, unstable=True)
        c, res, e = W.score(Q); pe, re = e
        wins = [(0, 22), (22, 60), (60, 120), (120, 190), (190, 260), (260, 330), (330, 392), (392, 421)]
        return dict(i=i, terms=terms, cost=c, spoon=res, early=[float(pe[:190].mean()), float(re[:190].mean())], late=[float(pe[190:].mean()), float(re[190:].mean())],
                    win=[[round(float(pe[a:b].mean()), 2), round(float(re[a:b].mean()), 1)] for a, b in wins])
    except Exception as ex: return dict(i=i, terms=terms, error=repr(ex)[:200])
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--actor', required=True); ap.add_argument('--out', required=True); ap.add_argument('--spec', required=True)
    ap.add_argument('--table'); ap.add_argument('--workers', type=int, default=16); ap.add_argument('--ep', type=int, default=35)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = dict(ep=a.ep, actor=str(Path(a.actor).resolve()), table=a.table, epd=None, failp=99.0, faild=999.0)
    spec = json.load(open(a.spec)); spec = [[]] + spec; cfg['need_base'] = any('roll' in tm for c in spec for tm in c); rows = []; t0 = time.monotonic()
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context('spawn'), initializer=init, initargs=(cfg,)) as pool:
        for r in pool.map(ev, list(enumerate(spec)), chunksize=1):
            rows.append(r); print(json.dumps(r), flush=True)
    rows.sort(key=lambda r: r.get('cost', 1e9)); json.dump(dict(cfg=cfg, rows=rows, elapsed=time.monotonic() - t0), open(out / 'grid.json', 'w'), indent=0)
    print('SUMMARY (sorted by cost)')
    for r in rows[:30]: print(round(r.get('cost', 1e9), 3), r.get('spoon'), 'early', np.round(r.get('early', [0, 0]), 2).tolist(), 'late', np.round(r.get('late', [0, 0]), 2).tolist(), r.get('win'), json.dumps(r['terms']))
if __name__ == '__main__': main()
