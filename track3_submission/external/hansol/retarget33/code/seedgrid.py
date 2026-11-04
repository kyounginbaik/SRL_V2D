"""K33-HG: robot-side wrist-residual smoothing grid through the spin. Replaces the right-hand table channels over [a,b] by a smoothstep between the values at a and b, full rollouts.
usage: E31_TGT=.. seedgrid.py VAR TABLE OUT WORKERS"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import sys, json, numpy as np, multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mresH as mres
V, TB, OUT, WK = sys.argv[1], sys.argv[2], Path(sys.argv[3]), int(sys.argv[4]); OUT.mkdir(parents=True, exist_ok=True)
CFG = dict(ep=33, actor='/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamH13a/runs/save_ep33_b1/actor', meshes='/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates/G18_ep33_H13b1/episode_000033/meshes/episode_000033', table=TB, freeze=None)
E = None
def init(): global E; E = mres.Env(CFG)
def job(a):
    name, tb = a; r = E.evaluate(tb, keep=True)
    if 'err' not in r: return dict(name=name, fail=True)
    e = mres.L.per_frame(33, E.poses(r['Q']), E.reg, E.tgt, want_mppe=True); d = e['wooden_spoon'][1]; c = e['blue_cup']
    return dict(name=name, cup=[float(c[0].mean()), float(c[1].mean())], spoon=[float(e['wooden_spoon'][0].mean()), float(d.mean())], peak=float(d[155:205].max()), pkf=int(155 + d[155:205].argmax()), m160=float(d[160:200].mean()), mppe=float(e['_mppe'].mean()), cupmax=float(c[0].max()))
if __name__ == '__main__':
    t0 = np.load(TB, allow_pickle=False)['table']; jobs = [('base', t0)]
    def ramp(a, b, sl):
        t = t0.copy(); u = (np.arange(a, b + 1) - a) / (b - a); s = mres.sm(u)[:, None]; t[a:b + 1, sl] = (1 - s) * t0[a, sl] + s * t0[b, sl]; return t
    for a in (140, 146, 150, 154, 158, 162):
        for b in (170, 174, 178, 184, 190, 196, 204):
            jobs.append((f'rot_{a}_{b}', ramp(a, b, slice(3, 6)))); jobs.append((f'pr_{a}_{b}', ramp(a, b, slice(0, 6))))
    with ProcessPoolExecutor(max_workers=WK, mp_context=mp.get_context('spawn'), initializer=init) as p: res = list(p.map(job, jobs, chunksize=1))
    res = [r for r in res if not r.get('fail')]; res.sort(key=lambda r: r['peak'] + r['m160'])
    (OUT / 'grid.json').write_text(json.dumps(res, indent=1)); tbs = dict(jobs)
    for r in res[:8]: np.savez_compressed(OUT / f"{r['name']}.npz", table=tbs[r['name']])
    print('BASE', [r for r in res if r['name'] == 'base'])
    for r in res[:15]: print(json.dumps({k: (round(v, 2) if isinstance(v, float) else (([round(x, 2) for x in v]) if isinstance(v, list) else v)) for k, v in r.items()}))
