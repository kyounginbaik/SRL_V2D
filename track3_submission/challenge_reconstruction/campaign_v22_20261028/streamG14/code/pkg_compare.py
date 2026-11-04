"""Official-consistency check: historical official packages (+ v27) scored against target_final_v10_g12 vs target_final_v11_g14 (G14 hedge)
on the 12 episodes the hedge changed. If no package changes SP/MP outcome on any episode, the official counts are explained exactly
as under v10 (G12/G13 calibration unchanged). Also prints v27 per-object deltas (part 3)."""
import json, glob, os
O = '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG14/out/pkg'
EPS = [1, 5, 6, 8, 10, 15, 16, 19, 20, 32, 33, 38]
PK = ['first', 'pf6', 'v13', 'v21C', 'v22b', 'v24', 'v25', 'v26', 'v27']
OFF = {'first': (6, 13), 'pf6': (6, 13), 'v13': (4, 11), 'v21C': (4, 9), 'v22b': (3, 7), 'v24': (1, 4), 'v25': (0, 4), 'v26': (0, 4)}
def get(p, t, e):
    f = f'{O}/{p}_{t}/ep{e}.json'
    if not os.path.exists(f): return None
    return json.load(open(f))[str(e)]
res = {}; flips = []
print('package  | SP/MP fails on the 12 changed eps: truth=v10 -> truth=v11 | official (all 20)')
for p in PK:
    c = {}
    for t in ('v10', 'v11'):
        sp = mp = 0; miss = 0
        for e in EPS:
            r = get(p, t, e)
            if r is None or r.get('scores') is None or r['scores'].get('MP-SR') is None: miss += 1; continue
            sp += r['scores']['SP-SR'] < 1; mp += r['scores']['MP-SR'] < 1
        c[t] = (sp, mp, miss)
    for e in EPS:
        a, b = get(p, 'v10', e), get(p, 'v11', e)
        if a and b and a.get('scores') and b.get('scores') and a['scores'].get('MP-SR') is not None and b['scores'].get('MP-SR') is not None:
            if (a['scores']['SP-SR'], a['scores']['MP-SR']) != (b['scores']['SP-SR'], b['scores']['MP-SR']):
                flips.append((p, e, a['scores'], b['scores'], a['objects'], b['objects']))
    res[p] = c
    print(f"{p:8s} | v10 SP {c['v10'][0]} MP {c['v10'][1]} -> v11 SP {c['v11'][0]} MP {c['v11'][1]} (missing {c['v10'][2]}/{c['v11'][2]}) | {OFF.get(p, '-')}")
print('\noutcome changes v10 -> v11:', len(flips))
for f in flips: print('  ', f[0], 'ep', f[1], f[2], '->', f[3], f[4], '->', f[5])
print('\nv27 per episode (object pos cm / ori deg; MPPE, AUC) vs v10 -> vs v11:')
v27 = {}
for e in EPS:
    a, b = get('v27', 'v10', e), get('v27', 'v11', e)
    if not (a and b): print('  ep', e, 'missing'); continue
    v27[e] = dict(v10=a, v11=b)
    s = '  '.join(f"{n}: {a['objects'][n][0]:.2f}/{a['objects'][n][1]:.1f} -> {b['objects'][n][0]:.2f}/{b['objects'][n][1]:.1f}" for n in a['objects'])
    print(f"  ep{e:2d} {s} | MPPE {a['scores']['MPPE']:.3f} -> {b['scores']['MPPE']:.3f} AUC {a['scores']['AUC']:.3f} -> {b['scores']['AUC']:.3f} | MP {a['scores']['MP-SR']:.0f}->{b['scores']['MP-SR']:.0f}")
json.dump(dict(counts=res, flips=[list(map(str, f)) for f in flips], v27=v27), open(f'{O}/../pkg_compare.json', 'w'), indent=1, default=str)
