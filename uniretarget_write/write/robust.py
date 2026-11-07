"""Are the passing fixed grips real? Rerun the top grips under changed physics; report jamming, finger torque, and survival. Each grip was found under nominal physics only."""
import json, os, sys, numpy as np, mujoco, multiprocessing as mp, planner as PL, wmetric as W
R = [json.loads(l) for l in open('planner_real.jsonl')]; P = sorted([r for r in R if r['PASS'] and r.get('pen_depth_cm', 9) <= .3], key=lambda r: r['tip_med_cm'])
print('all passes', len(P)); base = PL.setup
COND = {'nominal': {}, 'friction x0.7': dict(fr=.7), 'friction x0.5': dict(fr=.5), 'friction x1.3': dict(fr=1.3), 'finger kp x2': dict(kp=2.), 'pen mass x2': dict(mass=2.), 'finger kp x0.5': dict(kp=.5), 'board +3 mm': dict(bd=.003), 'board -3 mm': dict(bd=-.003), 'torque limit 2 Nm': dict(tl=2.), 'torque limit 0.5 Nm': dict(tl=.5), 'grip error 1mm .02rad a': dict(gn=(.001, .02, 1)), 'grip error 1mm .02rad b': dict(gn=(.001, .02, 2)), 'grip error 2mm .05rad a': dict(gn=(.002, .05, 3)), 'grip error 2mm .05rad b': dict(gn=(.002, .05, 4)), 'grip error 4mm .1rad': dict(gn=(.004, .1, 5))}
def run(a):
    r, c = a; o = COND[c]
    def setup():
        m, d, b = base()
        PG = {i for i in range(m.ngeom) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or '').startswith('right_object_')}; BG = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'board')
        if 'fr' in o:
            for i in range(m.npair):
                if (m.pair_geom1[i] in PG) != (m.pair_geom2[i] in PG) and BG not in (m.pair_geom1[i], m.pair_geom2[i]): m.pair_friction[i, :2] *= o['fr']
        if 'mass' in o:
            pb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'right_object'); m.body_mass[pb] *= o['mass']; m.body_inertia[pb] *= o['mass']
        if 'kp' in o: m.actuator_gainprm[6:, 0] *= o['kp']; m.actuator_biasprm[6:, 1] *= o['kp']
        if 'bd' in o: m.geom_pos[BG] += W.N * o['bd']
        if 'tl' in o: m.actuator_forcelimited[6:] = 1; m.actuator_forcerange[6:] = [-o['tl'], o['tl']]
        return m, mujoco.MjData(m), b
    PL.setup = setup; PL.GN = o.get('gn')
    x = PL.rollout((r['f'], r['squeeze'], r['servo'], r['press'], r['seed'])); return dict(c=c, seed=r['seed'], PASS=bool(x['PASS'] and x.get('pen_depth_cm', 9) <= .3), rawpass=x['PASS'], fail=x.get('fail'), tip=x.get('tip_med_cm'), onb=x.get('on_board'), held=x.get('held'), depth=x.get('pen_depth_cm'), contacts=x.get('contacts'), tau=x.get('tau_med'), tau99=x.get('tau_p99'))
if __name__ == '__main__':
    top = P[:100]; jobs = [(r, c) for r in top for c in COND]
    with mp.Pool(int(os.environ.get('SLURM_CPUS_PER_TASK', 8))) as pool: X = pool.map(run, jobs, chunksize=4)
    json.dump(X, open('robust_real.json', 'w'))
    for c in COND:
        y = [x for x in X if x['c'] == c]; t = [x['tip'] for x in y if x['tip'] is not None]; print(f"{c:22s} pass+overlap {sum(x['PASS'] for x in y):3d}/{len(y)}, raw {sum(x['rawpass'] for x in y):3d}  median tip {np.median(t):5.2f}" )
    y = [x for x in X if x['c'] == 'nominal' and x['PASS']]; print('nominal rerun pass', len(y), 'overlap cm p50', np.median([x['depth'] for x in y]), 'contacts', np.median([x['contacts'] for x in y]), 'finger torque Nm', np.median([x['tau'] for x in y]), 'p99', np.median([x['tau99'] for x in y]))
    seeds = {}
    for x in X: seeds[x['seed']] = seeds.get(x['seed'], 0) + bool(x['PASS'])
    b = sorted(seeds.items(), key=lambda s: -s[1]); print('grips passing all', len(COND), 'conditions:', sum(v == len(COND) for _, v in b), 'most robust', b[:5])
