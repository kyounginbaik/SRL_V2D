"""G14 part 4: most informative MP-SR-only knockout uploads on top of v27.
State = (A, b32): A = the 3 non-ep32 episodes failing MP in v26 (= in v27, only ep32 changed), b32 = v27 ep32 passes MP.
Prior over A: joint posteriors from calib14_joint.py (G13 calibration DP, failure SET kept), restricted to |A|=3 + ep32,
mixture over the 8 settings (equal weight) and evaluated per setting for robustness. P(b32) = 0.7 (G12/H12: 0.6-0.8).
An upload of v27 with set S knocked out returns MP-SR = (20 - |S u F|)/20, F = A u ({32} if not b32): deterministic given the state,
so the expected information = entropy of the returned count.
usage: ko_plan.py [P32=0.7]"""
import sys, json, glob, itertools, math, collections
import numpy as np
O = '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG14/out'
P32 = float(sys.argv[1]) if len(sys.argv) > 1 else 0.7
files = sorted(glob.glob(f'{O}/joint_*.json'))
def load(f):
    J = collections.defaultdict(float)
    for s, p in json.load(open(f))['joint']:
        s = set(s)
        if 32 in s and len(s) == 4: J[frozenset(s - {32})] += p
    z = sum(J.values()); return {k: v / z for k, v in J.items()}
settings = {f.split('joint_')[1][:-5]: load(f) for f in files}
mix = collections.defaultdict(float)
for J in settings.values():
    for k, v in J.items(): mix[k] += v / len(settings)
CAND = sorted({e for J in settings.values() for k in J for e in k})
marg = {e: sum(v for k, v in mix.items() if e in k) for e in CAND}
def states(J):
    for A, p in J.items():
        yield A, True, p * P32
        yield A, False, p * (1 - P32)
def outcome(S, A, b):
    F = set(A) | (set() if b else {32}); return 20 - len(set(S) | F)
def H(dist):
    return -sum(p * math.log2(p) for p in dist.values() if p > 0)
def info1(S, J):
    d = collections.defaultdict(float)
    for A, b, p in states(J): d[outcome(S, A, b)] += p
    return H(d)
def H_state(J):
    d = collections.defaultdict(float)
    for A, b, p in states(J): d[(A, b)] += p
    return H(d)
def infoA(S, J, S2=None):
    """mutual information between the returned count(s) and the failure set A (ep32 status is a nuisance)."""
    d = collections.defaultdict(float); c = collections.defaultdict(lambda: collections.defaultdict(float)); pa = collections.defaultdict(float)
    for A, b, p in states(J):
        o = (outcome(S, A, b), outcome(S2, A, b) if S2 is not None else 0); d[o] += p; c[A][o] += p; pa[A] += p
    return H(d) - sum(pa[A] * H({o: v / pa[A] for o, v in c[A].items()}) for A in pa if pa[A] > 0)
def info2(S1, S2, J):
    d = collections.defaultdict(float)
    for A, b, p in states(J): d[(outcome(S1, A, b), outcome(S2, A, b))] += p
    return H(d)
pool = [e for e in CAND if marg[e] > 0.02]
print('candidates (mixture marginal P(fail in v26)):', {e: round(marg[e], 3) for e in pool}, ' P32 pass', P32)
print('state entropy (mixture) %.2f bits' % H_state(mix))
allS = [tuple(sorted(c)) for r in range(0, len(pool) + 2) for c in itertools.combinations(pool + [32], r)]
res1 = sorted(((info1(S, mix), min(info1(S, J) for J in settings.values()), S) for S in allS), reverse=True)
print('\nONE upload: top sets by expected information (bits; mixture / worst setting)')
for i, m, S in res1[:15]: print(f'  {i:.3f} / {m:.3f}  S={list(S)}')
print('  plain v27 (S={}) %.3f' % info1((), mix))
resA = sorted(((infoA(S, mix), min(infoA(S, J) for J in settings.values()), info1(S, mix), S) for S in allS), reverse=True)
print('\nONE upload ranked by information about the 3 failures only (bits; mixture / worst setting / total)')
for i, m, t, S in resA[:12]: print(f'  {i:.3f} / {m:.3f} / {t:.3f}  S={list(S)}')
topA = [S for *_, S in resA[:60]]
b2 = sorted(((infoA(a, mix, b), a, b) for x, a in enumerate(topA) for b in topA[x + 1:]), reverse=True)
print('\nTWO uploads (non-adaptive) ranked by information about the 3 failures')
for i, a, b in b2[:8]: print(f'  {i:.3f}  S1={list(a)}  S2={list(b)}')
# two uploads, non-adaptive (top 60 singles crossed) and adaptive (second chosen per first outcome)
top = [S for _, _, S in res1[:80]]
best2 = []
for a in range(len(top)):
    for b in range(a + 1, len(top)):
        best2.append((info2(top[a], top[b], mix), top[a], top[b]))
best2.sort(reverse=True)
print('\nTWO uploads, non-adaptive: top pairs (bits)')
for i, a, b in best2[:8]: print(f'  {i:.3f}  S1={list(a)}  S2={list(b)}')
def adaptive(S1, J):
    groups = collections.defaultdict(list)
    for A, b, p in states(J): groups[outcome(S1, A, b)].append((A, b, p))
    tot = 0.0; plan = {}
    for k, g in groups.items():
        pk = sum(p for _, _, p in g)
        best = (-1, None)
        for S2 in allS:
            d = collections.defaultdict(float)
            for A, b, p in g: d[outcome(S2, A, b)] += p / pk
            h = H(d)
            if h > best[0]: best = (h, S2)
        d1 = pk; tot += pk * best[0]; plan[k] = (round(pk, 3), list(best[1]), round(best[0], 3))
    return info1(S1, J) + tot, plan
ad = sorted(((adaptive(S, mix)[0], S) for S in top[:25]), reverse=True)
print('\nTWO uploads, adaptive: best first sets (total bits)')
for i, S in ad[:5]: print(f'  {i:.3f}  S1={list(S)}')
bestS1 = ad[0][1]; tot, plan = adaptive(bestS1, mix)
print('  plan for S1=%s:' % list(bestS1))
for k, v in sorted(plan.items()): print(f'    MP count {k}/20 (p={v[0]}): upload S2={v[1]} (+{v[2]} bits)')
# decoding table for the recommended single upload
def decode(S, J):
    tab = collections.defaultdict(lambda: collections.defaultdict(float))
    for A, b, p in states(J): tab[outcome(S, A, b)][(tuple(sorted(A)), b)] += p
    out = {}
    for k, d in sorted(tab.items()):
        z = sum(d.values()); post = {e: sum(v for (A, b), v in d.items() if e in A) / z for e in pool}
        out[k] = dict(p=round(z, 3), p32pass=round(sum(v for (A, b), v in d.items() if b) / z, 3), marg={e: round(x, 2) for e, x in post.items() if x > 0.01},
                      top=[(list(A), b, round(v / z, 3)) for (A, b), v in sorted(d.items(), key=lambda kv: -kv[1])[:4]])
    return out
rep = dict(oneA=[(i, m, t, list(S)) for i, m, t, S in resA[:30]], twoA=[(i, list(a), list(b)) for i, a, b in b2[:20]], P32=P32, settings=list(settings), mixture_marginals=marg, one=[(i, m, list(S)) for i, m, S in res1[:30]],
           two_nonadaptive=[(i, list(a), list(b)) for i, a, b in best2[:20]], two_adaptive_first=[(i, list(S)) for i, S in ad[:10]],
           adaptive_plan={str(k): v for k, v in plan.items()}, decode_best1={str(k): v for k, v in decode(res1[0][2], mix).items()})
json.dump(rep, open(f'{O}/ko_plan_p{P32:g}.json', 'w'), indent=1, default=str)
print('\ndecode table for S=%s' % list(res1[0][2]))
for k, v in decode(res1[0][2], mix).items(): print(' ', k, v)
