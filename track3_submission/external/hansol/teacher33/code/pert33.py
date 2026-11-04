"""T33: perturbation test of open-loop TEACHERS (reference.npz + result.json): servo rollouts of reference + team fcem_r noise
(lside33.noise_draws: all robot command columns, smoothed 0.25 s, seeds 777+j) at each LEVEL x N draws; pass = both objects < 3 cm / 30 deg
vs $E31_TGT; spoon drop / cup lost as lside33.drop_metrics. usage: pert33.py OUT.json LEVELS N TEACHER_DIR[:LABEL] ..."""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import sys, json, numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
sys.path.insert(0, str(Path(__file__).resolve().parent)); import lside33 as LS
def job(a):
    te, lv, j = a; b = LS.Base(te); Q = b.Q0.copy()
    if lv > 0: Q = Q + LS.noise_draws(b.m, b.n, lv, j + 1)[j]
    r = LS.drop_metrics(b.m, b.rollout(Q)); r.update(level=lv, draw=j); return te, r
def main():
    out, lvs, N = Path(sys.argv[1]), [float(x) for x in sys.argv[2].split(',')], int(sys.argv[3]); tes = [a.partition(':') for a in sys.argv[4:]]
    jobs = [(t, 0.0, 0) for t, _, _ in tes] + [(t, lv, j) for t, _, _ in tes for lv in lvs for j in range(N)]
    with ProcessPoolExecutor(max_workers=min(16, len(jobs)), mp_context=mp.get_context('spawn')) as pool: res = list(pool.map(job, jobs))
    rep = {}
    for t, _, lab in tes:
        rs = [r for tt, r in res if tt == t]; nom = [r for r in rs if r['level'] == 0][0]; row = dict(nominal=nom, levels={})
        for lv in lvs:
            p = [r for r in rs if r['level'] == lv]
            row['levels'][str(lv)] = dict(n=len(p), passed=sum(r['passed'] for r in p), spoon_drop=sum(r['spoon_drop'] for r in p), cup_lost=sum(r['cup_lost'] for r in p),
                                         spoon_med=[round(float(np.median([r['summary']['wooden_spoon'][0] for r in p])), 2), round(float(np.median([r['summary']['wooden_spoon'][1] for r in p])), 1)],
                                         r_fingers=round(float(np.mean([r['r_fingers'] for r in p])), 2))
        rep[lab or t] = row
        print(f'== {lab or t}: nominal {nom["summary"]} pass {nom["passed"]} drop {nom["spoon_drop"]} cup_lost {nom["cup_lost"]} R fingers {nom["r_fingers"]:.2f}, spoon-palm max {nom["spoon_palm_max_cm"]:.1f} cm')
        for lv, v in row['levels'].items(): print(f'   noise {1000*float(lv):.1f} mm/mrad: pass {v["passed"]}/{v["n"]}, spoon drop {v["spoon_drop"]}/{v["n"]}, cup lost {v["cup_lost"]}/{v["n"]}, median spoon {v["spoon_med"]}, R fingers {v["r_fingers"]}')
    out.write_text(json.dumps(rep, indent=1))
if __name__ == '__main__': main()
