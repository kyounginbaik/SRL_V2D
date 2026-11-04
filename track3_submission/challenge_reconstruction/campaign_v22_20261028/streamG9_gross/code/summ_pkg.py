import json, glob, sys, os
G = '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG9_gross/out/pkg'
labels = sys.argv[1:] or sorted(os.listdir(G))
eps = [1, 3, 5, 6, 8, 10, 14, 15, 16, 19, 20, 25, 28, 30, 32, 33, 35, 36, 37, 38]
for L in labels:
    tot = dict(mp=0, sp=0, mppe=0, auc=0, n=0); rows = []
    for e in eps:
        p = f'{G}/{L}/ep{e}.json'
        if not os.path.exists(p): rows.append(f'{e}:--'); continue
        d = json.load(open(p))[str(e)]; s = d['scores']
        tot['mp'] += s['MP-SR']; tot['sp'] += s['SP-SR']; tot['mppe'] += s['MPPE']; tot['auc'] += s['AUC']; tot['n'] += 1
        flag = ('' if s['MP-SR'] == 1 else 'm') + ('' if s['SP-SR'] == 1 else 'S')
        rows.append(f"{e}{flag}:" + ','.join(f"{k[:6]}{v[0]}/{v[1]}" for k, v in d['objects'].items()))
    n = max(tot['n'], 1)
    print(f"== {L}: n={tot['n']} MP {tot['mp']/n:.2f} SP {tot['sp']/n:.2f} MPPE {tot['mppe']/n:.2f} AUC {tot['auc']/n:.3f}")
    for r in rows: print('  ', r)
