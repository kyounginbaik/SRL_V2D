"""Variant B: 20-clip public/held-out summary per variant folder (r8_base = fam_base; B1; B2). Scores come from the unchanged scorer."""
import json, glob, sys, numpy as np
W = '/home/hansol/v2d/outputs/variants_20261030/B_work'
res = {}
for var in sys.argv[1:] or ['fam_base', 'B1', 'B2']:
    clips = {}
    for f in sorted(glob.glob(f'{W}/canonical/*/episode_*/exports/{var}/public_score.json')):
        r = json.load(open(f)); clips[f.split('/')[-5] + '/' + str(r['episode'])] = r
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
base = res.get('fam_base')
if base:
    for var in res:
        if var == 'fam_base': continue
        d = [(c, res[var]['per_clip'][c]['MPPE'] - base['per_clip'][c]['MPPE'], res[var]['per_clip'][c]['AUC'] - base['per_clip'][c]['AUC']) for c in base['per_clip'] if c in res[var]['per_clip']]
        print(var, 'per-clip dMPPE/dAUC (changed only):', [(c, round(a, 3), round(b, 4)) for c, a, b in d if abs(a) > 1e-9])
json.dump(res, open(f'{W}/public_summary.json', 'w'), indent=1)
