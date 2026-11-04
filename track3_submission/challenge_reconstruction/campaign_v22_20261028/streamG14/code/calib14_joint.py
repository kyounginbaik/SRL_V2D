"""G12 task B: per-episode official calibration over ALL scored packages.
Each package p has an official (#SP fails, #MP fails). For each episode e and candidate truth label t, the kit-faithful pass
indicators of p are known (out/pkg/<p>_<target>/ep<e>.json). Labels: o = ours (scan-in-v9: our v9 trajectories, scan meshes
scorer-side so the twin/registration behaviour is official-like), c21, c27 (coworker tracks as hypotheses), i21/i27 = truth
between ours and compose (SP fails only if both fail, MP fails if either fails), plus extra per-episode labels (e.g. ep32 G12 track).
Exact DP over episodes of the joint failure-count vector; likelihood = prior(label) x prod_p exp(-|count - official| * PEN).
Reports per-episode label posteriors and P(fail MP in v26) / P(fail SP in v26).
usage: calib.py [PEN=3] [PRIOR_O=0.6]"""
import sys, json, glob, os, itertools, collections, math
import numpy as np
G = '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG12'; G13 = '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG13'; TAG = sys.argv[3] if len(sys.argv) > 3 else 'img'
OUTJ = sys.argv[4] if len(sys.argv) > 4 else None
EPS = [1, 3, 5, 6, 8, 10, 14, 15, 16, 19, 20, 25, 28, 30, 32, 33, 35, 36, 37, 38]
OFF = {'first': (6, 13), 'pf6': (6, 13), 'v13': (4, 11), 'v21C': (4, 9), 'v22b': (3, 7), 'v24': (1, 4), 'v25': (0, 4), 'v26': (0, 4)}
PK = list(OFF)
PEN = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
PO = float(sys.argv[2]) if len(sys.argv) > 2 else 0.6
EXTRA = json.load(open(f'{G}/out/calib_extra.json')); EXTRA.update(json.load(open(f'{G13}/out/calib_extra_g13.json')))
PRIOR_EP = json.load(open(f'{G}/out/calib_prior.json')); PRIOR_EP.update(json.load(open(f'{G13}/out/calib_prior_{TAG}.json')))


def get(p, t, e):
    fn = f'{G}/out/pkg/{p}_{t}/ep{e}.json'
    if not os.path.exists(fn): return None
    s = json.load(open(fn))[str(e)].get('scores')
    if not s or s.get('MP-SR') is None: return None
    return (int(s['SP-SR'] < 1), int(s['MP-SR'] < 1))


tab = {}
missing = []
for e in EPS:
    tab[e] = {}
    base = {t: {p: get(p, t, e) for p in PK} for t in ('scanv9', 'c21', 'c27')}
    for t, lab in (('scanv9', 'o'), ('c21', 'c21'), ('c27', 'c27')):
        if all(base[t][p] is not None for p in PK): tab[e][lab] = {p: base[t][p] for p in PK}
        else: missing.append((e, t, [p for p in PK if base[t][p] is None]))
    for c in ('c21', 'c27'):
        if 'o' in tab[e] and c in tab[e]:
            tab[e]['i' + c[1:]] = {p: (tab[e]['o'][p][0] & tab[e][c][p][0], tab[e]['o'][p][1] | tab[e][c][p][1]) for p in PK}
    for lab, d in EXTRA.get(str(e), {}).items():
        if all(p in d for p in PK): tab[e][lab] = {p: tuple(d[p]) for p in PK}
if missing: print('missing', missing[:10], '...', len(missing))


def prior(e, lab):
    if str(e) in PRIOR_EP and lab in PRIOR_EP[str(e)]: return PRIOR_EP[str(e)][lab]
    if lab == 'o': return PO
    return (1 - PO) / max(1, len(tab[e]) - 1)



# G14: joint posterior over the SET of v26 MP failures (state = count vector + failure set)
def lik(st):
    dev = sum(abs(st[2 * i] - OFF[p][0]) + abs(st[2 * i + 1] - OFF[p][1]) for i, p in enumerate(PK))
    return math.exp(-PEN * dev)
order = [e for e in EPS if tab[e]]
states = {((0,) * (2 * len(PK)), frozenset()): 1.0}
iv = PK.index('v26')
for e in order:
    nxt = collections.defaultdict(float)
    for (st, fs), w in states.items():
        for lab, d in tab[e].items():
            v = list(st)
            for i, p in enumerate(PK): v[2 * i] += d[p][0]; v[2 * i + 1] += d[p][1]
            if sum(max(0, v[2 * i] - OFF[p][0]) + max(0, v[2 * i + 1] - OFF[p][1]) for i, p in enumerate(PK)) > 4: continue
            f2 = fs | {e} if d['v26'][1] else fs
            nxt[(tuple(v), f2)] += w * prior(e, lab)
    states = dict(nxt)
J = collections.defaultdict(float)
for (st, fs), w in states.items(): J[fs] += w * lik(st)
Z = sum(J.values()); J = {k: v / Z for k, v in J.items()}
top = sorted(J.items(), key=lambda kv: -kv[1])
print(f'PEN {PEN} PO {PO} TAG {TAG}: {len(J)} failure sets; top:')
for k, v in top[:12]: print(f'  {v:.3f} {sorted(k)}')
import collections as C2
size = C2.Counter();
for k, v in J.items(): size[len(k)] += v
print('  set size dist', {k: round(v, 3) for k, v in sorted(size.items())})
if OUTJ: json.dump(dict(pen=PEN, po=PO, tag=TAG, joint=[[sorted(k), v] for k, v in top]), open(OUTJ, 'w'))
