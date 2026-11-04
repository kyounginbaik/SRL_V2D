"""Which per-episode 'truth' labels (ours=final_v6 via scan-in-v6 target, or compose) reproduce the official SP/MP failure
counts of v21C (4/9), v22b (3/7), v24 (1/4)? Enumerate labels over the disputed episodes; report posterior label frequency."""
import json, itertools, sys, numpy as np
G = '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG9_gross/out/pkg'
EPS = [1, 3, 5, 6, 8, 10, 14, 15, 16, 19, 20, 25, 28, 30, 32, 33, 35, 36, 37, 38]
OFF = {'v21C': (4, 9), 'v22b': (3, 7), 'v24': (1, 4)}
def get(lbl, e):
    s = json.load(open(f'{G}/{lbl}/ep{e}.json'))[str(e)]['scores']; return (s['SP-SR'] < 1, s['MP-SR'] < 1)
for comp in sys.argv[1:] or ['c27', 'c21']:
    tab = {p: {e: {'ours': get(f'{p}_scanv6', e), 'comp': get(f'{p}_{comp}' if comp in ('c21', 'c27') else f'{p}_{comp}', e)} for e in EPS} for p in OFF}
    disputed = [e for e in EPS if any(tab[p][e]['ours'] != tab[p][e]['comp'] for p in OFF)]
    print(f'== comp={comp}; disputed eps: {disputed}')
    for p in OFF: print('  ', p, 'ours-fails SP', [e for e in EPS if tab[p][e]['ours'][0]], 'MP', [e for e in EPS if tab[p][e]['ours'][1]], '| comp-fails SP', [e for e in EPS if tab[p][e]['comp'][0]], 'MP', [e for e in EPS if tab[p][e]['comp'][1]])
    for tolmp in (0, 1, 2):
        cnt = {e: [0, 0, 0] for e in disputed}; n = 0; best = []
        for lab in itertools.product([0, 1, 2], repeat=len(disputed)):
            L = dict(zip(disputed, lab)); ok = True
            for p, (sp, mp) in OFF.items():
                def st(e):
                    o, c = tab[p][e]['ours'], tab[p][e]['comp']; l = L.get(e, 0)
                    return o if l == 0 else c if l == 1 else (o[0] and c[0], o[1] or c[1])
                fs = sum(st(e)[0] for e in EPS); fm = sum(st(e)[1] for e in EPS)
                if fs != sp or abs(fm - mp) > tolmp: ok = False; break
            if ok:
                n += 1
                for e in disputed: cnt[e][L[e]] += 1
                if len(best) < 6: best.append({e: 'cm'[L[e] - 1] for e in disputed if L[e]})
        print(f'  tol MP +-{tolmp}: {n} consistent assignments; P(comp right) per ep:', {e: '/'.join(str(round(x / n, 2)) for x in cnt[e]) for e in disputed} if n else '-', '| examples (comp-right eps):', best)
