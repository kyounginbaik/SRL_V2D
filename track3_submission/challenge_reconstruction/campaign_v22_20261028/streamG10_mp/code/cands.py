"""Official-constraint calibration for one spoon episode: candidate truths = our final_v6 with compose-implied spoon poses
adopted on subsets of the disagreement segments (and variants); for each, the kit-faithful time-mean errors of the v24 and
v25 achieved rollouts. usage: cands.py EP [COMP=c27]"""
import sys, json, itertools, numpy as np
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG10_mp/code')
from g10lib import *
from implied import implied
EP = int(sys.argv[1]); COMP = sys.argv[2] if len(sys.argv) > 2 else 'c27'
B6 = f'{RR}/selected20_final_v6'
subs = {pk: load_sub(PKG[pk], EP) for pk in ('v24', 'v25')}
names = subs['v24'][0]
refT6, refM, d6 = load_bundle(B6, EP, names)
sc = {pk: Scorer(EP, names, subs[pk][2], refM) for pk in subs}
si = names.index('wooden_spoon')
imp = {}
for mode in ('rot', 'pose'):
    w, _, ang, _ = implied(EP, 'wooden_spoon', COMP, mode); imp[mode] = w
nf = len(refT6[si])
# segments
segs = []; t = 0
while t < nf:
    if ang[t] > 90:
        a = t
        while t < nf and ang[t] > 90: t += 1
        segs.append([a, t - 1])
    else: t += 1
mg = []
for sg in segs:
    if mg and sg[0] - mg[-1][1] <= 4: mg[-1][1] = sg[1]
    else: mg.append(sg)
grown = []
for a, b in mg:
    while a > 0 and ang[a - 1] > 45: a -= 1
    while b < nf - 1 and ang[b + 1] > 45: b += 1
    if grown and a <= grown[-1][1] + 1: grown[-1][1] = max(b, grown[-1][1])
    else: grown.append([a, b])
print('segments (>90 merged, grown to 45):', grown, 'frames', nf, flush=True)
res = []
def run(label, spoonT, cupT=None):
    refT = list(refT6); refT[si] = spoonT
    if cupT is not None: refT[1 - si] = cupT
    row = dict(label=label)
    for pk in subs:
        s, pos, ori = sc[pk].summary(subs[pk][1], refT)
        row[pk] = s
    res.append(row)
    print(label, '|', ' | '.join(f"{pk} {row[pk]['objects']} MP{int(row[pk]['MP'])} SP{int(row[pk]['SP'])} m{row[pk]['sp_mean_ori']}" for pk in subs), flush=True)
    return row
run('v6', refT6[si])
for mode in ('rot', 'pose'):
    run(f'{COMP}_{mode}_all', imp[mode])
    ks = range(1, len(grown) + 1) if len(grown) <= 8 else [1, len(grown) - 1]
    for k in ks:
        for sub in itertools.combinations(range(len(grown)), k):
            T = refT6[si].copy()
            for i in sub:
                a, b = grown[i]; T[a:b + 1] = imp[mode][a:b + 1]
            run(f'{COMP}_{mode}_segs' + '+'.join(f'{grown[i][0]}-{grown[i][1]}' for i in sub), T)
json.dump(dict(segments=grown, rows=res), open(f'{C}/streamG10_mp/out/cands_ep{EP}_{COMP}.json', 'w'), indent=1)
np.savez(f'{C}/streamG10_mp/out/implied_ep{EP}_{COMP}.npz', rot=imp['rot'], pose=imp['pose'], ang=ang)
