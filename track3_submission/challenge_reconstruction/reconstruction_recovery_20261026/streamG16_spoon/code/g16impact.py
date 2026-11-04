"""G16: kit-faithful effect of a constant spoon body-roll correction (GT-free hypothesis; our meshes are the reference).
For EP and ROLL (deg, about the long PCA axis through the surface centroid; same convention as g16scan):
 - delta between the adopted v12b track and the corrected one (time-mean pos cm / ori deg per object, MP/SP of v12b vs v13)
 - packages v24/v25/v26 (official ep32 outcomes known) and v28 (current) scored against: v10_g12 (RGB clock, the world the
   official calibration used), v10_g12+roll, v12b, v12b+roll.
usage: g16impact.py EP ROLL"""
import sys, json, numpy as np
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG10_mp/code')
from g10lib import load_sub, load_bundle, Scorer, roll_axis, body_roll, RR
EP = int(sys.argv[1]); ROLL = float(sys.argv[2])
G = f'{RR}/streamG16_spoon'
S = '/data1/SRL_V2D/challenge_submissions_SRL'
PKG = {'v24': f'{S}/track3_v24_20261028/all_20_episodes', 'v25': f'{S}/track3_v25_20261028/all_20_episodes',
       'v26': f'{S}/track3_v26_20261029/all_20_episodes', 'v28': f'{S}/track3_v28_20261029/all_20_episodes'}
subs = {k: load_sub(p, EP) for k, p in PKG.items()}
names = subs['v28'][0]; si = names.index('wooden_spoon')
T12, M12, _ = load_bundle(f'{RR}/selected20_final_v12b_g15', EP, names)
T10, _, _ = load_bundle(f'{RR}/selected20_final_v10_g12', EP, names)
c, ax = roll_axis(*M12[si])
def rolled(Ts):
    o = list(Ts); o[si] = body_roll(Ts[si], c, ax, ROLL); return o
truths = {'v10': T10, 'v10r': rolled(T10), 'v12b': T12, 'v13': rolled(T12)}
out = dict(ep=EP, roll=ROLL, names=names)
sc = Scorer(EP, names, [m for m in M12], M12)
s, pos, ori = sc.summary(T12, truths['v13'])
out['delta_v12b_vs_v13'] = s; out['delta_pf_ori_p90'] = {n: float(np.percentile(ori[:, i], 90)) for i, n in enumerate(names)}
print(EP, ROLL, 'v12b vs v13:', s, flush=True)
for pk, (nm, Ts, cands) in subs.items():
    assert nm == names
    scp = Scorer(EP, names, cands, M12)
    out[pk] = {}
    for tn, T in truths.items():
        r = scp.summary(Ts, T)[0]; out[pk][tn] = r
        print(f'  {pk} vs {tn}: {r}', flush=True)
json.dump(out, open(f'{G}/out/impact_ep{EP}_r{ROLL:g}.json', 'w'), indent=1)
