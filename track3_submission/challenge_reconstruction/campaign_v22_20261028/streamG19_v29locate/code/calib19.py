"""G19: official-consistency calibration with v29 added (G12 calib.py / G13 calib13 / G14 calib14_joint extended).
Packages: first, pf6, v13, v21C, v22b, v24, v25, v26, v29 with official (#SP fails, #MP fails).
Per episode e, candidate truth labels t give kit-faithful (SP fail, MP fail) of every package (out/tab/<label>/ep<e>.json, tab19.py).
Labels: o = scan-in-v14 (our final v14_g17 trajectories, public scans / c27 spoon surrogate scorer-side; OBASE=scanv9 for G12's o),
c21, c27, i21, i27 (between ours and compose: SP fails iff both fail, MP fails iff either fails), fro (+ ifro) where EXT,
ep32: o32_v10 / o32_v13 (scan-in-v10 consensus RGB clock, scan-in-v13 roll-bias) + o (= scan-in-v12b consensus retimed),
ep3: o3_v12b / o3_v10 (no ep3 roll fix) + o (= v13 roll fix), eps 19/25/30: o_tw (G12 twin wildcard for the old packages).
Exact DP over episodes; state = (count vector, F26, F29); likelihood exp(-PEN * sum |count - official|).
usage: calib19.py PEN PRIOR_O PRIORSET(img|dedup0.4|dedup0.6|dedup0.8|none) LABELSET(g12|ext) OBASE(scanv14|scanv9) OUT.json"""
import sys, json, os, collections, math, itertools
S = '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG19_v29locate'
G12 = '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG12'; G13 = '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG13'
EPS = [1, 3, 5, 6, 8, 10, 14, 15, 16, 19, 20, 25, 28, 30, 32, 33, 35, 36, 37, 38]
OFF = {'first': (6, 13), 'pf6': (6, 13), 'v13': (4, 11), 'v21C': (4, 9), 'v22b': (3, 7), 'v24': (1, 4), 'v25': (0, 4), 'v26': (0, 4), 'v29': (0, 3)}
PK = list(OFF)
PEN = float(sys.argv[1]); PO = float(sys.argv[2]); PSET = sys.argv[3]; LSET = sys.argv[4]; OBASE = sys.argv[5]; OUTJ = sys.argv[6]
EXCL = set(int(x) for x in os.environ.get('G19_EXCLUDE_PKG_EPS', '').split(',') if x)  # unused hook


def load(lab, e):
    fn = f'{S}/out/tab/{lab}/ep{e}.json'
    if not os.path.exists(fn): return None
    d = json.load(open(fn))
    if not all(p in d and 'mp' in d[p] for p in PK): return None
    return {p: (d[p]['sp'], d[p]['mp']) for p in PK}


tab = {}; miss = []
for e in EPS:
    t = {}
    o = load(OBASE, e)
    if o is None: miss.append((e, OBASE)); continue
    t['o'] = o
    comp = ['c21', 'c27'] + (['fro'] if (LSET == 'ext' or e in (20, 33)) else [])
    for c in comp:
        d = load(c, e)
        if d is None: miss.append((e, c)); continue
        t[c] = d
        if c != 'fro' or LSET == 'ext':
            t['i' + c] = {p: (o[p][0] & d[p][0], o[p][1] | d[p][1]) for p in PK}
    if e == 32:
        for lab in ('scanv10', 'scanv13'):
            d = load(lab, 32)
            if d is not None: t['o32_' + lab[4:]] = d
    if e == 3:
        for lab in ('scanv12b', 'scanv10'):
            d = load(lab, 3)
            if d is not None: t['o3_' + lab[4:]] = d
    if e in (19, 25, 30):  # G12 twin wildcard: old-package outcomes from G12, new packages as o
        g = json.load(open(f'{G12}/out/calib_extra.json'))[str(e)]['o_tw']
        t['o_tw'] = {p: tuple(g[p]) if p in g and p not in ('v24', 'v25', 'v26') else o[p] for p in PK}
    tab[e] = t
if miss: print('missing', miss)

# ep32: hard constraint = known official pattern v24 SP fail, v25/v26 SP pass + MP fail (G10/G12)
for lab in list(tab[32]):
    d = tab[32][lab]
    if not (d['v24'][0] == 1 and d['v25'] == (0, 1) and d['v26'] == (0, 1)): tab[32].pop(lab)
print('ep32 labels kept', {l: ''.join(f'{d[p][0]}{d[p][1]}' for p in PK) for l, d in tab[32].items()})

PR = {}
if PSET.startswith('dedup'): PR = json.load(open(f'{G13}/out/calib_prior_{PSET}.json'))
elif PSET == 'img': PR = json.load(open(f'{G13}/out/calib_prior_img.json'))
PR = {k: dict(v) for k, v in PR.items()}
for k in ('19', '25', '30'): PR.setdefault(k, {}).update({'o': 0.3, 'o_tw': 0.3})


def prior(e, lab):
    d = PR.get(str(e), {})
    if d and lab in d: return d[lab]
    if d:  # unlisted labels share what is left
        rest = [l for l in tab[e] if l not in d]; left = max(0.0, 1 - sum(v for l, v in d.items() if l in tab[e]))
        return left / max(1, len(rest)) if rest else 0.0
    if e == 32 or e == 3:
        os_ = [l for l in tab[e] if l == 'o' or l.startswith('o32_') or l.startswith('o3_')]
        if lab in os_: return PO / len(os_)
        return (1 - PO) / max(1, len(tab[e]) - len(os_))
    if lab == 'o': return PO
    return (1 - PO) / max(1, len(tab[e]) - 1)


iv26, iv29 = PK.index('v26'), PK.index('v29')
states = {((0,) * (2 * len(PK)), frozenset(), frozenset()): 1.0}
for e in EPS:
    nxt = collections.defaultdict(float)
    for (st, f26, f29), w in states.items():
        for lab, d in tab[e].items():
            pw = prior(e, lab)
            if pw <= 0: continue
            v = list(st)
            for i, p in enumerate(PK): v[2 * i] += d[p][0]; v[2 * i + 1] += d[p][1]
            if sum(max(0, v[2 * i] - OFF[p][0]) + max(0, v[2 * i + 1] - OFF[p][1]) for i, p in enumerate(PK)) > 5: continue
            nxt[(tuple(v), f26 | ({e} if d['v26'][1] else set()), f29 | ({e} if d['v29'][1] else set()))] += w * pw
    states = dict(nxt)


def dev(st): return sum(abs(st[2 * i] - OFF[p][0]) + abs(st[2 * i + 1] - OFF[p][1]) for i, p in enumerate(PK))


J = collections.defaultdict(float); best = 99
for (st, f26, f29), w in states.items():
    dv = dev(st); best = min(best, dv); J[(f26, f29)] += w * math.exp(-PEN * dv)
Z = sum(J.values()); J = {k: v / Z for k, v in J.items()}
m29 = collections.defaultdict(float); m26 = collections.defaultdict(float); fixed = collections.defaultdict(float); newf = collections.defaultdict(float)
J29 = collections.defaultdict(float); size29 = collections.defaultdict(float)
for (f26, f29), p in J.items():
    for e in f29: m29[e] += p
    for e in f26: m26[e] += p
    for e in f26 - f29: fixed[e] += p
    for e in f29 - f26: newf[e] += p
    J29[tuple(sorted(f29))] += p; size29[len(f29)] += p
top29 = sorted(J29.items(), key=lambda kv: -kv[1])
print(f'PEN {PEN} PO {PO} PSET {PSET} LSET {LSET} OBASE {OBASE}: min count deviation {best}')
print('  P(v29 MP fail):', {e: round(m29[e], 3) for e in sorted(m29, key=lambda x: -m29[x]) if m29[e] > 0.005})
print('  P(v26 MP fail):', {e: round(m26[e], 3) for e in sorted(m26, key=lambda x: -m26[x]) if m26[e] > 0.005})
print('  P(fail v26 -> pass v29):', {e: round(fixed[e], 3) for e in sorted(fixed, key=lambda x: -fixed[x]) if fixed[e] > 0.005})
print('  P(pass v26 -> fail v29):', {e: round(newf[e], 3) for e in sorted(newf, key=lambda x: -newf[x]) if newf[e] > 0.005})
print('  |F29| dist', {k: round(v, 3) for k, v in sorted(size29.items())})
for k, v in top29[:10]: print(f'  {v:.3f} F29={list(k)}')
json.dump(dict(pen=PEN, po=PO, pset=PSET, lset=LSET, obase=OBASE, min_dev=best, labels={e: {l: ''.join(f'{d[p][0]}{d[p][1]}' for p in PK) for l, d in tab[e].items()} for e in tab},
               p29=m29, p26=m26, fixed=fixed, newfail=newf, size29=size29,
               joint=[[sorted(a), sorted(b), p] for (a, b), p in sorted(J.items(), key=lambda kv: -kv[1]) if p > 1e-5]),
          open(OUTJ, 'w'), indent=1, default=str)
