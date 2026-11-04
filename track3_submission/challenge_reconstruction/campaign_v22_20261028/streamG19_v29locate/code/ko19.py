"""G19 knockout / swap planner on top of v29.
Prior over full labelings = FFBS samples of calib19s.py (out/samp/*.json), equal-weight mixture over settings.
A design assigns each suspect episode either 'keep' (the v29 component: deterministic outcome under the sampled label, exactly as
in the calibration) or a build-v2 candidate component (exact replay + portable archive) whose MP fail probability under a label is
soft: P(pass) = prod_obj Phi((3 - pos)/SP) Phi((30 - ori)/SO) from the kit-faithful time-mean errors vs that label's truth
(out/tab/K_<truth>/ep*.json); i-labels (truth between ours and compose): fail if either fails.
Returned MP-SR x 20 = 20 - #fails. Score = mutual information I(count; F29) (F29 = the v29 failing set) + expected MP.
usage: ko19.py OUT.json [SP=0.5] [SO=5] [SUSPECTS=1,3,19,20,32,33,35,36]"""
import sys, json, glob, os, math, itertools, collections
import numpy as np, functools
print = functools.partial(print, flush=True)
from scipy.stats import norm
S = '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG19_v29locate'
CD = '/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates'; OV = '/home/exaFLOPs/v22_overflow_20261028/common_v2_candidates'
OUT = sys.argv[1]; SP = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5; SO = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
SUS = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else '1,3,19,20,32,33,35,36').split(',')]
FILES = sorted(glob.glob(f'{S}/out/samp/s_*.json'))
if os.environ.get('G19_SAMP_FILTER'): FILES = [f for f in FILES if os.environ['G19_SAMP_FILTER'] in f]


def exact_ok(tag, ep):
    n = f'episode_{ep:06d}'; d = f'{CD}/{tag}/{n}' if os.path.isdir(f'{CD}/{tag}/{n}') else f'{OV}/{tag}/{n}'
    try: c = json.load(open(f'{d}/completed.json'))
    except Exception: return False
    if c.get('exact_previous_physical_states_actions') is not True: return False
    from pathlib import Path
    leg = Path(c['legacy_component']); return (leg.parents[1] / 'portable_archives' / f'{n}_portable.tar.gz').is_file()


KT = {}
def ktab(truth, ep):
    k = (truth, ep)
    if k not in KT:
        fn = f'{S}/out/tab/K_{truth}/ep{ep}.json'; KT[k] = json.load(open(fn)) if os.path.exists(fn) else {}
    return KT[k]


def pfail_src(tag, ep, truth):
    d = ktab(truth, ep).get(tag)
    if not d or 'obj' not in d: return None
    pp = 1.0
    for pm, om in d['obj']: pp *= norm.cdf((3 - pm) / SP) * norm.cdf((30 - om) / SO)
    return 1 - pp


def label_src(lab, obase, ep):
    """truth source(s) of a calibration label for candidate scoring; returns list (i-labels: two)."""
    o = obase
    if lab in ('o', 'o_tw'): return [o]
    if lab in ('c21', 'c27', 'fro'): return [lab]
    if lab in ('ic21', 'ic27', 'ifro'): return [o, lab[1:]]
    if lab == 'o32_v10': return ['scanv10']
    if lab == 'o32_v13': return ['scanv13']
    if lab == 'o3_v12b': return ['scanv12b']
    if lab == 'o3_v10': return ['scanv10']
    raise KeyError(lab)


def pfail(tag, ep, lab, obase):
    srcs = label_src(lab, obase, ep); ps = [pfail_src(tag, ep, s) for s in srcs]
    if any(p is None for p in ps): return None
    q = 1.0
    for p in ps: q *= (1 - p)
    return 1 - q


# ---- load samples (mixture), aggregated by (obase, suspect labels, other-episode v29 fails, F29) ----
EPS = [1, 3, 5, 6, 8, 10, 14, 15, 16, 19, 20, 25, 28, 30, 32, 33, 35, 36, 37, 38]
agg = collections.defaultdict(float); KEEP = {}
for f in FILES:
    d = json.load(open(f)); tot = sum(c for _, c in d['samples']); tb = d['tab']
    for labs, c in d['samples']:
        L = dict(zip(d['eps'], labs))
        F = frozenset(e for e in EPS if tb[str(e)][L[e]]['v29'][1])
        b = sum(1 for e in F if e not in SUS)
        key = (d['obase'], tuple(L[e] for e in SUS), b, F)
        agg[key] += c / tot / len(FILES)
        for e in SUS: KEEP[(d['obase'], e, L[e])] = tb[str(e)][L[e]]['v29'][1]
rows = [(w, dict(zip(SUS, k[1])), k[0], k[2], k[3]) for k, w in agg.items()]
print('settings', len(FILES), 'aggregated rows', len(rows))
W = np.array([r[0] for r in rows]); W /= W.sum()
F29 = [r[4] for r in rows]
marg = {e: float(sum(w for w, F in zip(W, F29) if e in F)) for e in EPS}
print('mixture P(v29 fail)', {e: round(v, 3) for e, v in sorted(marg.items(), key=lambda kv: -kv[1]) if v > 0.005})
fkeys = sorted(set(F29), key=lambda F: -sum(w for w, G in zip(W, F29) if G == F)); fid = {F: i for i, F in enumerate(fkeys)}
FI = np.array([fid[F] for F in F29])
J29 = collections.Counter()
for w, F in zip(W, F29): J29[tuple(sorted(F))] += w
print('top F29', [(k, round(v, 3)) for k, v in J29.most_common(8)])

# ---- options per suspect ----
cands = {}
for ep in SUS:
    tags = [t for t in ktab('scanv14', ep) if 'obj' in ktab('scanv14', ep)[t] and exact_ok(t, ep)]
    cands[ep] = tags
# per sample, per ep, per option: P(fail)
labs_needed = collections.defaultdict(set)
for w, L, ob, b, F in rows:
    for ep in SUS: labs_needed[ep].add((L[ep], ob))
PF = {}  # (ep, tag) -> {(lab, obase): p}
for ep in SUS:
    for t in cands[ep]:
        m = {}
        for lab, ob in labs_needed[ep]:
            p = pfail(t, ep, lab, ob)
            if p is None: m = None; break
            m[(lab, ob)] = p
        if m is not None: PF[(ep, t)] = m
    cands[ep] = [t for t in cands[ep] if (ep, t) in PF]
# option matrix: for ep, option index -> vector over rows of P(fail); option 0 = keep (deterministic)
opt_names = {}; OPT = {}
for ep in SUS:
    keep = np.array([float(KEEP[(r[2], ep, r[1][ep])]) for r in rows])
    names = ['keep']; vecs = [keep]
    for t in cands[ep]:
        names.append(t); vecs.append(np.array([PF[(ep, t)][(r[1][ep], r[2])] for r in rows]))
    opt_names[ep] = names; OPT[ep] = np.stack(vecs)
base = np.array([r[3] for r in rows])


def H(p):
    p = p[p > 1e-12]; return float(-(p * np.log2(p)).sum())


def pmf(design, idx=None):
    """design: {ep: option index}; returns [n_rows, 21] count-of-fails PMF."""
    n = len(rows) if idx is None else len(idx)
    P = np.zeros((n, 21)); b = base if idx is None else base[idx]
    P[np.arange(n), b] = 1.0
    for ep in SUS:
        q = OPT[ep][design.get(ep, 0)]
        q = q if idx is None else q[idx]
        P2 = P * (1 - q)[:, None]; P2[:, 1:] += P[:, :-1] * q[:, None]; P = P2
    return P


def evaluate(design, idx=None, w=None):
    w = W if w is None else w; w = w / w.sum(); P = pmf(design, idx); fi = FI if idx is None else FI[idx]
    pc = (w[:, None] * P).sum(0)
    hc_f = 0.0
    for k in np.unique(fi):
        m = fi == k; wk = w[m].sum()
        if wk > 0: hc_f += wk * H((w[m][:, None] * P[m]).sum(0) / wk)
    emp = float(((20 - np.arange(21)) * pc).sum()) / 20
    return H(pc) - hc_f, H(pc), emp, pc


# per-episode option pruning: keep, plus per ep the options that are most distinct in their fail-vectors (dedupe near-identical)
def prune(ep, k=int(os.environ.get('G19_K', '1'))):
    V = OPT[ep]; keep_i = [0]
    # rank options by marginal information when swapped alone
    sc = []
    for i in range(1, len(V)):
        I, _, emp, _ = evaluate({ep: i}); sc.append((I, emp, i))
    sc.sort(reverse=True)
    for I, emp, i in sc:
        if all(np.abs(V[i] - V[j]).max() > 0.05 for j in keep_i): keep_i.append(i)
        if len(keep_i) >= k + 1: break
    # always add the best expected-MP option (fix candidate) and the strongest knockout
    best_fix = max(range(len(V)), key=lambda i: -float((W * V[i]).sum()))
    best_ko = max(range(1, len(V)), key=lambda i: float(V[i].min())) if len(V) > 1 else 0
    for i in (best_fix, best_ko):
        if i not in keep_i: keep_i.append(i)
    return keep_i, sc


P_OPTS = {}; single = {}
FORCE = {int(a.split(':')[0]): a.split(':')[1].split(',') for a in os.environ.get('G19_OPTS', '').split(';') if a}
for ep in SUS:
    ki, sc = prune(ep)
    for t in FORCE.get(ep, []):
        if t in opt_names[ep] and opt_names[ep].index(t) not in ki: ki.append(opt_names[ep].index(t))
    P_OPTS[ep] = ki
    single[ep] = [(opt_names[ep][i], round(I, 3), round(emp, 4), round(float(OPT[ep][i].min()), 3), round(float((W * OPT[ep][i]).sum()), 3)) for I, emp, i in sc[:8]]
    print(f'ep{ep}: options', [opt_names[ep][i] for i in ki])
    for row in single[ep][:5]: print('    single swap', row)

if os.environ.get('G19_EVAL'):
    EV = json.load(open(os.environ['G19_EVAL'])); outE = {}
    def decode2(d):
        P = pmf(d); tab = {}
        for c in range(21):
            pw = W * P[:, c]; z = pw.sum()
            if z < 1e-3: continue
            m = collections.Counter()
            for k, F in enumerate(F29): m[tuple(sorted(F))] += pw[k]
            mg = {e: round(float(sum(pw[k] for k, F in enumerate(F29) if e in F) / z), 2) for e in EPS}
            tab[20 - c] = dict(p=round(float(z), 3), marg={e: v for e, v in mg.items() if v > 0.02}, top=[(list(k), round(v / z, 3)) for k, v in m.most_common(4)])
        return tab
    for name, des in EV.items():
        d = {int(e): opt_names[int(e)].index(t) for e, t in des.items()}
        I, Hc, emp, pc = evaluate(d); outE[name] = dict(design=des, I=I, EMP=emp, P_MP=[[ (20 - c) / 20, round(float(pc[c]), 3)] for c in range(21) if pc[c] > 1e-3], decode=decode2(d))
        print(name, 'I %.3f E[MP] %.4f' % (I, emp), [(x, y) for x, y in outE[name]['P_MP']])
        for k, v in sorted(outE[name]['decode'].items(), key=lambda kv: -kv[0]): print('   ', k, v)
    json.dump(dict(marg29=marg, top_F29=[(list(k), v) for k, v in J29.most_common(20)], H_F29=H(np.array(list(J29.values()))), evals=outE), open(OUT, 'w'), indent=1, default=str)
    raise SystemExit(0)
keep_I, keep_H, keep_E, keep_pc = evaluate({})
print('plain v29 re-upload: I %.3f, E[MP] %.4f' % (keep_I, keep_E))
ncomb = int(np.prod([len(P_OPTS[ep]) for ep in SUS]))
print('designs', ncomb)
res = []
if ncomb <= 70000:
    for dz in itertools.product(*[P_OPTS[ep] for ep in SUS]):
        d = {ep: i for ep, i in zip(SUS, dz) if i}
        I, Hc, emp, pc = evaluate(d); res.append((I, emp, d, pc))
else:
    raise SystemExit('too many designs; lower G19_K')
res.sort(key=lambda x: -x[0])
def dname(d): return {ep: opt_names[ep][i] for ep, i in d.items()}
print('\nTOP single uploads by I(count;F29)')
for I, emp, d, pc in res[:12]: print(f'  I {I:.3f} E[MP] {emp:.4f}', dname(d))
resE = sorted(res, key=lambda x: -(x[1]))
print('\nTOP single uploads by expected MP (with I)')
for I, emp, d, pc in resE[:8]: print(f'  E[MP] {emp:.4f} I {I:.3f}', dname(d))
# frontier: maximize I + lambda * 20*(E[MP]-0.85)
front = []
for lam in (0, 0.5, 1, 2, 4):
    b = max(res, key=lambda x: x[0] + lam * 20 * (x[1] - 0.85)); front.append((lam, b[0], b[1], dname(b[2])))
print('\nfrontier (lambda bits per expected extra pass)')
for f in front: print(' ', f)


def decode(d, w=None, idx=None):
    w = W if w is None else w; P = pmf(d); tab = {}
    for c in range(21):
        pw = w * P[:, c]; z = pw.sum()
        if z < 1e-4: continue
        m = collections.Counter()
        for k, F in enumerate(F29): m[tuple(sorted(F))] += pw[k]
        mg = {e: round(float(sum(pw[k] for k, F in enumerate(F29) if e in F) / z), 2) for e in EPS}
        tab[20 - c] = dict(p=round(float(z), 3), mp=(20 - c) / 20, marg={e: v for e, v in mg.items() if v > 0.02}, top=[(list(k), round(v / z, 3)) for k, v in m.most_common(4)])
    return tab


def adaptive(d1, top2=300):
    P1 = pmf(d1); tot = 0.0; plan = {}
    cand2 = [x[2] for x in res[:top2]] + [x[2] for x in resE[:50]]
    for c in range(21):
        w = W * P1[:, c]; z = w.sum()
        if z < 1e-3: continue
        best = (-1, None, None)
        for d2 in cand2:
            I2, _, e2, _ = evaluate(d2, w=w)
            if I2 > best[0]: best = (I2, d2, e2)
        tot += z * best[0]; plan[20 - c] = dict(p=round(float(z), 3), second=dname(best[1]), I2=round(best[0], 3), E2=round(best[2], 4))
    return tot, plan


ad = []
for I, emp, d, pc in res[:15] + resE[:5]:
    t, plan = adaptive(d); ad.append((I + t, I, emp, d, plan))
ad.sort(key=lambda x: -x[0])
print('\nTWO uploads adaptive (total bits about F29)')
for tot, I, emp, d, plan in ad[:5]: print(f'  total {tot:.3f} (first {I:.3f}, E[MP] {emp:.4f})', dname(d))
H29 = H(np.array([v for v in J29.values()]))
print('entropy of F29 %.3f bits' % H29)
rep = dict(settings=[os.path.basename(f) for f in FILES], SP=SP, SO=SO, suspects=SUS, marg29=marg, top_F29=[(list(k), v) for k, v in J29.most_common(20)], H_F29=H29,
           plain=dict(I=keep_I, EMP=keep_E), single_swaps=single,
           top_info=[dict(I=I, EMP=emp, design=dname(d), decode=decode(d)) for I, emp, d, pc in res[:10]],
           top_emp=[dict(I=I, EMP=emp, design=dname(d), decode=decode(d)) for I, emp, d, pc in resE[:5]],
           frontier=front,
           adaptive=[dict(total=t, I1=I, EMP1=emp, first=dname(d), plan=plan, decode1=decode(d)) for t, I, emp, d, plan in ad[:5]])
json.dump(rep, open(OUT, 'w'), indent=1, default=str)
