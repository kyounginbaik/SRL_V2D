import json, glob, collections, os
S = '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG19_v29locate'
fs = sorted(f for f in glob.glob(f'{S}/out/calib/c_*.json'))
groups = {'all': fs, 'o=scan-in-v14': [f for f in fs if 'scanv14' in f], 'o=scan-in-v9': [f for f in fs if 'scanv9' in f]}
res = {}
for g, L in groups.items():
    p29 = collections.defaultdict(float); fx = collections.defaultdict(float); nf = collections.defaultdict(float); J3 = collections.defaultdict(float); per = {}
    for f in L:
        d = json.load(open(f)); w = 1 / len(L)
        for e, v in d['p29'].items(): p29[int(e)] += w * v
        for e, v in d['fixed'].items(): fx[int(e)] += w * v
        for e, v in d['newfail'].items(): nf[int(e)] += w * v
        for a, b, p in d['joint']:
            if len(b) == 3: J3[tuple(b)] += w * p
        per[os.path.basename(f)[2:-5]] = {int(e): round(v, 2) for e, v in sorted(d['p29'].items(), key=lambda kv: -kv[1]) if v > 0.05}
    res[g] = dict(n=len(L), p29={e: round(v, 3) for e, v in sorted(p29.items(), key=lambda kv: -kv[1]) if v > 0.005},
                  fixed_v26_to_v29={e: round(v, 3) for e, v in sorted(fx.items(), key=lambda kv: -kv[1]) if v > 0.005},
                  new_fail_v29={e: round(v, 3) for e, v in sorted(nf.items(), key=lambda kv: -kv[1]) if v > 0.005},
                  top3=[(list(k), round(v, 3)) for k, v in sorted(J3.items(), key=lambda kv: -kv[1])[:10]], per_setting=per)
    print(g, res[g]['n']); print('  P(v29 fail)', res[g]['p29']); print('  fixed', res[g]['fixed_v26_to_v29']); print('  newfail', res[g]['new_fail_v29']); print('  top3', res[g]['top3'][:8])
json.dump(res, open(f'{S}/out/calib_summary.json', 'w'), indent=1)
