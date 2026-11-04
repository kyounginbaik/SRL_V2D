"""Official-outcome likelihood of candidate ep32 truths under hidden-GT noise. For each candidate target, N realisations of
smooth random body-frame rotation noise (sigma deg, ~10-frame correlation) + position noise (sigma_p cm) are applied to both
objects; the kit-faithful v24/v25 outcomes are recomputed. Reports P(v24 SP fail & v25 SP pass & v25 MP fail) = P(official).
usage: noise_mc.py EP SIGMA_DEG SIGMA_CM N TARGET_LABEL=DIR ..."""
import sys, json, numpy as np
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG10_mp/code')
from g10lib import *
from scipy.spatial.transform import Rotation as Rot
EP, SIG, SIGP, N = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
tg = dict(a.split('=', 1) for a in sys.argv[5:])
subs = {pk: load_sub(PKG[pk], EP) for pk in ('v24', 'v25')}; names = subs['v24'][0]
_, rm0 = load_target(f'{C}/target_final_v6', EP, names)
sc = {pk: Scorer(EP, names, subs[pk][2], rm0) for pk in subs}
rng = np.random.default_rng(7)
def smooth_noise(nf, sig, corr=10):
    w = rng.normal(size=(nf + 4 * corr, 3)); k = np.exp(-0.5 * (np.arange(-2 * corr, 2 * corr + 1) / corr) ** 2); k /= np.sqrt((k ** 2).sum())
    x = np.stack([np.convolve(w[:, i], k, 'same') for i in range(3)], 1)[2 * corr:2 * corr + nf]
    return x * sig
out = {}
for lab, td in tg.items():
    refT, _ = load_target(td, EP, names); nf = len(refT[0]); hits = 0; rows = []
    for it in range(N):
        T2 = []
        for i, T in enumerate(refT):
            rv = np.radians(smooth_noise(nf, SIG)); dp = smooth_noise(nf, SIGP / 100.)
            X = T.copy(); X[:, :3, :3] = T[:, :3, :3] @ Rot.from_rotvec(rv).as_matrix(); X[:, :3, 3] += dp; T2.append(X)
        r = {pk: sc[pk].summary(subs[pk][1], T2)[0] for pk in subs}
        ok = (not r['v24']['SP']) and r['v25']['SP'] and (not r['v25']['MP'])
        hits += ok; rows.append(dict(v24=r['v24']['objects'], v24m=r['v24']['sp_mean_ori'], v25=r['v25']['objects'], ok=ok))
    out[lab] = dict(p_official=hits / N, mean_v24_spoon=float(np.mean([x['v24']['wooden_spoon'][1] for x in rows])), mean_v25_spoon=float(np.mean([x['v25']['wooden_spoon'][1] for x in rows])),
                    mean_v24_m=float(np.mean([x['v24m'] for x in rows])))
    print(lab, SIG, SIGP, json.dumps(out[lab]), flush=True)
json.dump(out, open(f'{C}/streamG10_mp/out/noise_mc_ep{EP}_s{SIG:g}.json', 'w'), indent=1)
