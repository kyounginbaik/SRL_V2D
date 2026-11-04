"""Public/held-out scores of export folders vs a base export (our mirror; GT only inside the unchanged scorer).
usage: compare14.py BASE_EXPORT EXPORT... [-v]   e.g. compare14.py g15_camrt A14_g8 A14_s32
Writes A_work/compare14_<EXPORT>_vs_<BASE>.json."""
import sys, json
from pathlib import Path
import numpy as np
W = Path('/home/hansol/v2d/outputs/variants_20261030/A_work')
args = [a for a in sys.argv[1:] if a != '-v']; verbose = '-v' in sys.argv
base, tags = args[0], args[1:]
clips = sorted(list((W / 'pub' / 'public').glob('episode_*')) + list((W / 'pub' / 'public_heldout').glob('episode_*')))
load = lambda root, tag: json.loads(f.read_text()) if (f := root / 'exports' / tag / 'public_score.json').exists() else None
for tag in [base] + tags:
    rows, pc, miss = [], {}, 0
    for root in clips:
        r, b = load(root, tag), load(root, base)
        if r is None or b is None:
            miss += 1; continue
        pc[f'{root.parent.name}/{root.name}'] = dict(r['public_scores'], base=b['public_scores'])
        for n, e in r['public_errors'].items():
            be = b['public_errors'][n]
            rows.append((root.parent.name, int(root.name[-6:]), n, e['position_cm'], e['orientation_deg'], be['position_cm'], be['orientation_deg']))
    if not rows:
        print(tag, 'no scores'); continue

    def summ(rows, v):
        a = np.array([x[3:] for x in rows])
        return dict(clips=len(v), missing=miss, mpsr=sum(x['MP-SR'] for x in v), spsr=sum(x['SP-SR'] for x in v), pos=a[:, 0].mean(), ori=a[:, 1].mean(),
                    mppe=np.mean([x['MPPE'] for x in v]), auc=np.mean([x['AUC'] for x in v]), base_pos=a[:, 2].mean(), base_ori=a[:, 3].mean(),
                    base_mppe=np.mean([x['base']['MPPE'] for x in v]), base_auc=np.mean([x['base']['AUC'] for x in v]),
                    n_pos_worse=int((a[:, 0] > a[:, 2] + .05).sum()), n_pos_better=int((a[:, 0] < a[:, 2] - .05).sum()),
                    max_pos_delta=float((a[:, 0] - a[:, 2]).max()), max_ori_delta=float((a[:, 1] - a[:, 3]).max()))
    show = lambda lab, s: print(f"{lab:16s} vs {base}: clips {s['clips']} MP {s['mpsr']:.0f} SP {s['spsr']:.0f} | pos {s['pos']:.3f} (base {s['base_pos']:.3f}) ori {s['ori']:.2f} ({s['base_ori']:.2f}) "
                                f"MPPE {s['mppe']:.3f} ({s['base_mppe']:.3f}) AUC {s['auc']:.4f} ({s['base_auc']:.4f}) | worse/better >.05cm {s['n_pos_worse']}/{s['n_pos_better']} max delta +{s['max_pos_delta']:.2f}cm +{s['max_ori_delta']:.2f}deg")
    s = summ(rows, list(pc.values())); show(f'{tag} (miss {miss})', s)
    split = {}
    for sp in ['public', 'public_heldout']:
        split[sp] = {k: float(x) for k, x in summ([x for x in rows if x[0] == sp], [x for k, x in pc.items() if k.split('/')[0] == sp]).items()}
        show(f'  {sp}', split[sp])
    if tag != base:
        out = {k: float(x) for k, x in s.items()}; out['base'] = base; out['split'] = split
        out['per_object'] = [dict(split=x[0], episode=x[1], object=x[2], pos_cm=x[3], ori_deg=x[4], base_pos_cm=x[5], base_ori_deg=x[6]) for x in rows]
        out['per_clip'] = pc
        (W / f'compare14_{tag}_vs_{base}.json').write_text(json.dumps(out, indent=1))
        if verbose:
            for x in rows:
                print(f"    {x[0][:8]:8s} {x[1]:2d} {x[2]:18s} {x[5]:.3f}/{x[6]:.2f} -> {x[3]:.3f}/{x[4]:.2f}")
