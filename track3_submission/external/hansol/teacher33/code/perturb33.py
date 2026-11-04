"""[T33 copy of retarget32 perturb32.py: ep 33, G18 ep33 meshes, P0/P1 = 3.0/14.0 s (ep33 hold); output OUT/perturb33.json] R32: grasp perturbation test (team streamV8 rob.py draws): for each actor (+ residual table) the nominal closed-loop rollout and N random tiny
right-wrist command perturbations (8 smooth knots over [P0, P1] s; std 1 mm wrist pos / 8 mrad wrist rot / 8 mrad finger+thumb squeeze;
the SAME draws for every actor). Reports nominal and perturbed spoon / cup errors vs $E31_TGT, fraction passing (< 3 cm and < 30 deg both
objects), fraction that lose the spoon (spoon mean > 10 cm). -> OUT/perturb33.json
usage: E31_TGT=.. perturb32.py OUT N WORKERS name=ACTOR[:TABLE] ..."""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import sys, json, numpy as np, multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mresH as mres
M = '/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates/G18_ep33_H13b1/episode_000033/meshes/episode_000033'
P0, P1, KN, STD = 3.0, 14.0, 8, 0.001
def ev(tb):
    r = mres.W.evaluate(tb)
    o = r.get('objects')
    return dict(cup=o['blue_cup'], spoon=o['wooden_spoon']) if o else dict(fail=str(r)[:120])
if __name__ == '__main__':
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); N, WK = int(sys.argv[2]), int(sys.argv[3]); res = {}
    rng = np.random.default_rng(3200); X = [rng.normal(size=KN * 8) * np.tile([STD] * 3 + [STD * 8] * 5, KN) for _ in range(N)]
    for arg in sys.argv[4:]:
        nm, spec = arg.split('=', 1); actor, _, table = spec.partition(':'); cfg = dict(ep=33, actor=actor, meshes=M, table=table or None)
        env = mres.Env(cfg); base = env.table0.copy(); jobs = [base] + [base + mres.knot_table(env, x, P0, P1, KN, 'right') for x in X]
        with ProcessPoolExecutor(max_workers=WK, mp_context=mp.get_context('spawn'), initializer=mres.init, initargs=(cfg,)) as p: rs = list(p.map(ev, jobs, chunksize=1))
        ok = [r for r in rs[1:] if 'fail' not in r]; sp = np.array([r['spoon'] for r in ok]); cp = np.array([r['cup'] for r in ok])
        pas = [(s[0] < 3 and s[1] < 30 and c[0] < 3 and c[1] < 30) for s, c in zip(sp, cp)]
        res[nm] = dict(actor=actor, table=table or None, scene=env.scene, nominal=rs[0], n=len(ok), n_fail=len(rs) - 1 - len(ok),
                       spoon_cm_median=round(float(np.median(sp[:, 0])), 2), spoon_cm_q25_q75=[round(float(np.quantile(sp[:, 0], .25)), 2), round(float(np.quantile(sp[:, 0], .75)), 2)],
                       spoon_deg_median=round(float(np.median(sp[:, 1])), 1), frac_pass=round(float(np.mean(pas)), 3), frac_spoon_lost=round(float(np.mean(sp[:, 0] > 10)), 3),
                       perturbed=[dict(spoon=[round(float(a), 2), round(float(b), 1)], cup=[round(float(c), 2), round(float(e), 1)]) for (a, b), (c, e) in zip(sp, cp)])
        print('PERTURB', nm, json.dumps({k: v for k, v in res[nm].items() if k != 'perturbed'}), flush=True)
        (out / 'perturb33.json').write_text(json.dumps(res, indent=1))
