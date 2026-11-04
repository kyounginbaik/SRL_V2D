"""Variant B on v14: public summaries from the unchanged scorer's public_score.json files in our mirror (base v14_base = g15_camrt).
Writes public_summary14.json, per_type14.json (retimed families vs v14_base) and rules14.json (the rule re-derived on the v14 clock,
for comparison with the shipped rule)."""
import json, glob, sys, numpy as np
from collections import defaultdict
sys.path.insert(0, '/home/hansol/v2d/outputs/variants_20261030/B_work/code'); import variants as V
BW = '/home/hansol/v2d/outputs/variants_20261030/B_work'; W = '/home/hansol/v2d/outputs/variants_20261030/B14_work'
VARS = ['v14_base', 'v14_B1', 'v14_B2', 'v14_B1n', 'v14_B2n', 'v14_B2cv']
# 1) re-scored base == team score (determinism / mirror check)
bad = [f for f in glob.glob(f'{BW}/canonical/*/episode_*/exports/v14_base/public_score.json')
       if json.load(open(f))['public_scores'] != json.load(open(f.replace('public_score.json', 'score_team.json')))['public_scores']]
print('v14_base re-scored == team g15_camrt score:', len(glob.glob(f'{BW}/canonical/*/episode_*/exports/v14_base/public_score.json')), 'files, mismatches', bad)
res = {}
for var in VARS:
    clips = {}
    for f in sorted(glob.glob(f'{BW}/canonical/*/episode_*/exports/{var}/public_score.json')):
        r = json.load(open(f)); clips[f.split('/')[-5] + '/' + str(r['episode'])] = r
    if len(clips) != 20: print('WARNING', var, len(clips), 'clips');
    objs = [(c, o, e['position_cm'], e['orientation_deg']) for c, r in clips.items() for o, e in r['public_errors'].items()]
    def agg(sel):
        cs = [c for c in clips if sel(c)]; ob = [x for x in objs if sel(x[0])]
        return dict(clips=len(cs), mp=int(sum(clips[c]['public_scores']['MP-SR'] for c in cs)), sp=int(sum(clips[c]['public_scores']['SP-SR'] for c in cs)),
                    pos=float(np.mean([x[2] for x in ob])), ori=float(np.mean([x[3] for x in ob])),
                    MPPE=float(np.mean([clips[c]['public_scores']['MPPE'] for c in cs])), AUC=float(np.mean([clips[c]['public_scores']['AUC'] for c in cs])),
                    worst_obj=max(ob, key=lambda x: max(x[2] / 3, x[3] / 30))[:4])
    res[var] = dict(all=agg(lambda c: True), public=agg(lambda c: c.startswith('public/')), heldout=agg(lambda c: c.startswith('public_heldout/')),
                    per_clip={c: dict(MPPE=r['public_scores']['MPPE'], AUC=r['public_scores']['AUC'], MP=r['public_scores']['MP-SR'], errors=r['public_errors']) for c, r in clips.items()})
    for k in ['all', 'public', 'heldout']:
        a = res[var][k]; print(f"{var:9s} {k:8s} n={a['clips']} MP {a['mp']}/{a['clips']} SP {a['sp']} pos {a['pos']:.4f} ori {a['ori']:.3f} MPPE {a['MPPE']:.4f} AUC {a['AUC']:.4f} worst {a['worst_obj']}")
base = res['v14_base']
for var in VARS[1:]:
    d = [(c, res[var]['per_clip'][c]['MPPE'] - base['per_clip'][c]['MPPE'], res[var]['per_clip'][c]['AUC'] - base['per_clip'][c]['AUC']) for c in base['per_clip'] if c in res[var]['per_clip']]
    ch = [(c, round(a, 3), round(b, 4)) for c, a, b in d if abs(a) > 1e-9]
    res[var]['per_clip_delta'] = ch; print(var, 'clips better/worse MPPE:', sum(a < 0 for _, a, _ in ch), '/', sum(a > 0 for _, a, _ in ch), ch)
    wo = [(c, o, round(e['position_cm'] - base['per_clip'][c]['errors'][o]['position_cm'], 3), round(e['orientation_deg'] - base['per_clip'][c]['errors'][o]['orientation_deg'], 2))
          for c in res[var]['per_clip'] for o, e in res[var]['per_clip'][c]['errors'].items()]
    res[var]['worst_object_deltas'] = sorted(wo, key=lambda x: -(x[2] / 3 + x[3] / 30))[:4]; print('   worst object deltas (cm, deg):', res[var]['worst_object_deltas'])
json.dump(res, open(f'{W}/public_summary14.json', 'w'), indent=1)
# 2) per-type table of the retimed families vs v14_base
rows = defaultdict(dict)
for pat in ['fam14_*', 'v14_base']:
    for f in glob.glob(f'{BW}/canonical/*/episode_*/exports/{pat}/public_score.json'):
        p = f.split('/'); fam = 'fam_base' if p[-2] == 'v14_base' else 'fam_' + p[-2][6:]
        for o, e in json.load(open(f))['public_errors'].items(): rows[(p[-5], int(p[-4][-6:]), o)][fam] = (e['position_cm'], e['orientation_deg'])
typ = defaultdict(list)
for k, v in rows.items(): typ[k[2]].append(v)
pt = {}
print('\nper type (retimed families vs v14_base): family n pos ori mp_fail | paired dpos dori wins')
for o, items in sorted(typ.items()):
    pt[o] = {}
    for fm in sorted({f for v in items for f in v}):
        pb = [(v[fm], v['fam_base']) for v in items if fm in v]; fail = sum(a[0] >= 3 or a[1] >= 30 for a, _ in pb)
        pt[o][fm] = dict(n=len(pb), pos_mean=float(np.mean([a[0] for a, _ in pb])), ori_mean=float(np.mean([a[1] for a, _ in pb])), mp_fail=fail,
                         dpos_vs_base=float(np.mean([a[0] - b[0] for a, b in pb])), dori_vs_base=float(np.mean([a[1] - b[1] for a, b in pb])), wins=sum(a[0] < b[0] and a[1] < b[1] + 1e-9 for a, b in pb))
        s = pt[o][fm]; print(f"{o:18s} {fm[4:]:20s} {s['n']:2d} {s['pos_mean']:6.2f} {s['ori_mean']:6.2f} {fail:2d} | {s['dpos_vs_base']:+6.3f} {s['dori_vs_base']:+7.2f} {s['wins']}/{s['n']}")
json.dump(dict(per_object={f'{k[0]}/{k[1]}/{k[2]}': v for k, v in rows.items()}, per_type=pt), open(f'{W}/per_type14.json', 'w'), indent=1)
b1, good = V.type_rules(pt); shipped = json.load(open(f'{BW}/rules.json'))
json.dump(dict(B1=b1, B2_good=good, same_as_shipped=dict(B1=b1 == shipped['B1'], B2={t: sorted(good.get(t, [])) == sorted(shipped['B2_good'].get(t, [])) for t in set(good) | set(shipped['B2_good'])})), open(f'{W}/rules14.json', 'w'), indent=1)
print('\nrule re-derived on v14 clock: B1', b1, '\nB2 good', good, '\nshipped B1', shipped['B1'], '\nshipped B2', shipped['B2_good'])
