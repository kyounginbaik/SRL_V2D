"""decode tables + A-only adaptive second upload for chosen knockout sets (uses ko_plan.py machinery)."""
import sys, json, collections, itertools, math
sys.argv = [sys.argv[0], '0.7']
src = open('ko_plan.py').read(); src = src[:src.find("pool = [e for e in CAND")]
exec(src)
pool = [e for e in CAND if marg[e] > 0.02]
allS = [tuple(sorted(c)) for r in range(0, len(pool) + 2) for c in itertools.combinations(pool + [32], r)]
def table(S, J=mix):
    tab = collections.defaultdict(lambda: collections.defaultdict(float))
    for A, b, p in states(J): tab[outcome(S, A, b)][(tuple(sorted(A)), b)] += p
    out = {}
    for k, d in sorted(tab.items()):
        z = sum(d.values()); pa = collections.defaultdict(float)
        for (A, b), v in d.items(): pa[A] += v / z
        out[k] = dict(p=round(z, 3), p32=round(sum(v for (A, b), v in d.items() if b) / z, 2),
                      marg={e: round(sum(v for A, v in pa.items() if e in A), 2) for e in pool},
                      topA=[(list(A), round(v, 3)) for A, v in sorted(pa.items(), key=lambda kv: -kv[1])[:3]])
    return out
def HA(d):
    return -sum(p * math.log2(p) for p in d.values() if p > 0)
def postA(S, k, J=mix):
    d = collections.defaultdict(float)
    for A, b, p in states(J):
        if outcome(S, A, b) == k: d[A] += p
    z = sum(d.values()); return {a: v / z for a, v in d.items()}
def infoA_given(S2, prior_states):
    d = collections.defaultdict(float); c = collections.defaultdict(lambda: collections.defaultdict(float)); pa = collections.defaultdict(float)
    for A, b, p in prior_states:
        o = outcome(S2, A, b); d[o] += p; c[A][o] += p; pa[A] += p
    return HA(d) - sum(pa[A] * HA({o: v / pa[A] for o, v in c[A].items()}) for A in pa if pa[A] > 0)
res = {}
pA = collections.defaultdict(float)
for A, b, p in states(mix): pA[A] += p
print('prior H(A) = %.2f bits' % HA(pA))
for S in [(3, 20, 32, 33), (1, 19, 35, 36), (1, 19, 32, 35, 36), (3, 32, 33, 35, 36)]:
    t = table(S); res[str(list(S))] = t
    print('\nS =', list(S))
    for k, v in t.items():
        print(f"  MP {k}/20 = {k/20:.2f}  p={v['p']}  P(ep32 pass)={v['p32']}  marg={ {e: x for e, x in v['marg'].items() if x > 0.03} }  top={v['topA']}")
# adaptive A-only plan after S1 = (3,20,32,33)
S1 = (3, 20, 32, 33); plan = {}; tot = 0
for k, v in table(S1).items():
    st = [(A, b, p) for A, b, p in states(mix) if outcome(S1, A, b) == k]; z = sum(p for *_, p in st); st = [(A, b, p / z) for A, b, p in st]
    best = max(((infoA_given(S2, st), S2) for S2 in allS), key=lambda x: x[0])
    plan[k] = dict(p=v['p'], S2=list(best[1]), bits=round(best[0], 3)); tot += v['p'] * best[0]
    print(f'  after S1 count {k}: second upload S2={list(best[1])} (+{best[0]:.2f} bits about A)')
print('  adaptive total about A: %.2f bits (S1 alone %.2f)' % (tot + infoA_given(S1, list(states(mix))), infoA_given(S1, list(states(mix)))))
json.dump(dict(tables=res, adaptive_after_3_20_32_33=plan), open('../out/ko_decode.json', 'w'), indent=1)
