"""Summarise eval hedge runs: per episode/object worst-case (max over hypotheses) predicted time-mean error before (v10) and after (hedge),
over admitted hypotheses (inlier fraction >= MINFRAC) and over all; time-mean move of the hedge vs v10.
usage: summarize_eval.py TAG [MINFRAC=0.8]"""
import sys, json, glob, numpy as np
TAG = sys.argv[1]; MF = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8
O = f'/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG14/out/eval_{TAG}'
rows = []
for f in sorted(glob.glob(f'{O}/ep*.json'), key=lambda x: int(x.split('ep')[-1][:-5])):
    d = json.load(open(f)); ep = d['ep']; srcs = [s for s, _ in d['info']['sources']]
    for n, oi in d['info']['objects'].items():
        fr = dict(zip(srcs, oi['inlier_frac_per_source']))
        hy = {t: v[n] for t, v in d['per_hyp'].items() if n in v}
        adm = [t for t in hy if fr.get(t, 0) >= MF]
        def worst(ts, k):
            if not ts: return (0.0, 0.0, '-')
            p = max(hy[t][k][0] for t in ts); o = max(hy[t][k][1] for t in ts)
            r = max(ts, key=lambda t: max(hy[t][k][0] / 3, hy[t][k][1] / 30))
            return (p, o, r)
        rows.append(dict(ep=ep, obj=n, move_cm=oi['move_cm_mean'], move_deg=oi['move_deg_mean'], admitted=adm,
                         adm_before=worst(adm, 'before'), adm_after=worst(adm, 'after'), all_before=worst(list(hy), 'before'), all_after=worst(list(hy), 'after'),
                         per_hyp={t: dict(before=[round(x, 2) for x in hy[t]['before']], after=[round(x, 2) for x in hy[t]['after']], inlier_frac=round(fr.get(t, 0), 2)) for t in hy}))
json.dump(rows, open(f'{O}/summary.json', 'w'), indent=1)
print(f'{"ep":>3} {"object":18s} {"move cm/deg":>12s} | {"admitted worst before -> after (cm/deg, hyp)":52s} | all-hyp worst before -> after   admitted')
for r in rows:
    ab, aa, lb, la = r['adm_before'], r['adm_after'], r['all_before'], r['all_after']
    flag = ' *' if r['move_cm'] > 0.5 or r['move_deg'] > 5 else ''
    print(f"{r['ep']:3d} {r['obj'][:18]:18s} {r['move_cm']:5.2f}/{r['move_deg']:5.1f}{flag:2s} | {ab[0]:5.2f}/{ab[1]:5.1f} -> {aa[0]:5.2f}/{aa[1]:5.1f} ({ab[2]:>18s}) | {lb[0]:6.2f}/{lb[1]:6.1f} -> {la[0]:6.2f}/{la[1]:6.1f}  {','.join(r['admitted'])}")
