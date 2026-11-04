"""G16: official-outcome likelihood of ep32 truth candidates (v12b vs v13_g16, also RGB-clock v10/v10+roll) under hidden-GT
noise (G10 noise_mc model: smooth body-frame rotation noise SIG deg + position noise SIGP cm on both objects).
Official ep32: v24 SP0 MP0, v25 SP1 MP0, v26 SP1 MP0. Reports P(all three) per truth.
usage: g16mc.py SIG SIGP N [ROLLS=comma list: extra truths v12b rolled by these deg]"""
import sys, json, numpy as np
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG10_mp/code')
from g10lib import load_sub, load_bundle, Scorer, roll_axis, body_roll, RR
from scipy.spatial.transform import Rotation as Rot
SIG, SIGP, N = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]); EP = 32
S = '/data1/SRL_V2D/challenge_submissions_SRL'
PKG = {'v24': f'{S}/track3_v24_20261028/all_20_episodes', 'v25': f'{S}/track3_v25_20261028/all_20_episodes',
       'v26': f'{S}/track3_v26_20261029/all_20_episodes', 'v28': f'{S}/track3_v28_20261029/all_20_episodes'}
OFF = {'v24': (False, False), 'v25': (True, False), 'v26': (True, False)}
subs = {k: load_sub(p, EP) for k, p in PKG.items()}; names = subs['v24'][0]; si = names.index('wooden_spoon')
T12, M, _ = load_bundle(f'{RR}/selected20_final_v12b_g15', EP, names); T13, _, _ = load_bundle(f'{RR}/selected20_final_v13_g16', EP, names)
T10, _, _ = load_bundle(f'{RR}/selected20_final_v10_g12', EP, names)
c, ax = roll_axis(*M[si]); T10r = list(T10); T10r[si] = body_roll(T10[si], c, ax, -15.)
truths = {'v12b': T12, 'v13': T13, 'v10': T10, 'v10r': T10r}
if len(sys.argv) > 4:
    truths = {}
    for r in sys.argv[4].split(','):
        t = list(T12); t[si] = body_roll(T12[si], c, ax, float(r)); truths[f'v12b_r{r}'] = t
sc = {k: Scorer(EP, names, subs[k][2], M) for k in subs}
rng = np.random.default_rng(16)
def sn(nf, sig, corr=10):
    w = rng.normal(size=(nf + 4 * corr, 3)); k = np.exp(-0.5 * (np.arange(-2 * corr, 2 * corr + 1) / corr) ** 2); k /= np.sqrt((k ** 2).sum())
    return np.stack([np.convolve(w[:, i], k, 'same') for i in range(3)], 1)[2 * corr:2 * corr + nf] * sig
out = {}
for lab, refT in truths.items():
    nf = len(refT[0]); hit = 0; v28mp = 0
    for it in range(N):
        T2 = []
        for T in refT:
            X = T.copy(); X[:, :3, :3] = T[:, :3, :3] @ Rot.from_rotvec(np.radians(sn(nf, SIG))).as_matrix(); X[:, :3, 3] += sn(nf, SIGP / 100.); T2.append(X)
        r = {k: sc[k].summary(subs[k][1], T2)[0] for k in subs}
        hit += all((r[k]['SP'], r[k]['MP']) == OFF[k] for k in OFF); v28mp += r['v28']['MP']
    out[lab] = dict(p_official=hit / N, p_v28_mp_pass=v28mp / N)
    print(lab, SIG, SIGP, out[lab], flush=True)
json.dump(out, open(f'{RR}/streamG16_spoon/out/mc_ep32_s{SIG:g}_p{SIGP:g}' + (f'_rolls{sys.argv[4]}' if len(sys.argv) > 4 else '') + '.json', 'w'), indent=1)
