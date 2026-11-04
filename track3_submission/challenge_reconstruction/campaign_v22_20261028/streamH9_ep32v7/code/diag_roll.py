"""H9 diag: apply the (time-continuous) rigid-hold rot ILC over [t0,t1] at several alphas/sigmas in one shot; per 0.25 s spoon error, wrist cmd vs achieved roll."""
import sys, json, numpy as np
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamH9_ep32v7/code')
import mres
from scipy.ndimage import gaussian_filter1d as gf
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
actor, meshes, t0, t1 = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
cfg = dict(ep=32, actor=actor, meshes=meshes)
E = None
def init():
    global E; E = mres.Env(cfg)
def job(tb):
    r = E.evaluate(tb, keep=True)
    if 'err' not in r: return None
    return dict(objects=r['objects'], sp=r['err']['wooden_spoon'][1], spp=r['err']['wooden_spoon'][0], Q=r['Q'])
if __name__ == '__main__':
    env = mres.Env(cfg); r0 = env.evaluate(env.table0, keep=True); Q, A = r0['Q'], r0['A']
    raw = env.ilc_correction(Q, A, 'wooden_spoon', 'right', t0, t1, 0.2)
    print('raw corr rot (deg) every 5 frames', np.degrees(raw[::5, 3:]).round(0).tolist())
    cands, meta = [], []
    for sg in (0.0, 0.05, 0.1):
        cs = gf(raw, sg * env.fps, axis=0, mode='nearest') if sg else raw.copy(); cs[env.t < t0 - 0.2] = 0
        for al in (0.5, 1.0, 1.5):
            for md, sl in (('rot', slice(3, 6)), ('full', slice(0, 6))):
                tb = env.table0.copy(); tb[:, sl] = al * cs[:, sl]; cands.append(tb); meta.append(dict(sigma=sg, alpha=al, mode=md))
    with ProcessPoolExecutor(18, mp_context=mp.get_context('spawn'), initializer=init) as pool: reps = list(pool.map(job, cands))
    wq = env.wq['right']; k0 = int(t0 * env.fps); k1 = min(env.T, int((t1 + 1) * env.fps))
    print('base', r0['objects'], 'spoon deg per 5f', r0['err']['wooden_spoon'][1][k0:k1:5].round(0).tolist())
    for m, r in zip(meta, reps):
        if r is None: print(m, 'unstable'); continue
        dq = np.degrees(r['Q'][:, wq[3:]] - Q[:, wq[3:]])
        print(m, {k: [round(x, 2) for x in v] for k, v in r['objects'].items()}, 'win deg', round(float(r['sp'][k0:k1].mean()), 1), 'pos', round(float(r['spp'][k0:k1].mean()), 2))
        print('   spoon deg/5f', r['sp'][k0:k1:5].round(0).tolist())
        print('   wrist dq x/5f', dq[k0:k1:5, 0].round(0).tolist())
