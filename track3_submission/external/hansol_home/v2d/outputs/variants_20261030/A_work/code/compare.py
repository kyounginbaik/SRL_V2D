"""Public/held-out score table of A_<tag> variants vs r8_base (our mirror; GT only inside the unchanged scorer).
usage: compare.py [-v] TAG..."""
import sys, json
from pathlib import Path
import numpy as np
W = Path('/home/hansol/v2d/outputs/variants_20261030/A_work/pub')
verbose = '-v' in sys.argv
tags = [t for t in sys.argv[1:] if t != '-v']
clips = sorted(list((W / 'public').glob('episode_*')) + list((W / 'public_heldout').glob('episode_*')))
STORED = Path('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/canonical')


def load(root, tag):
    f = root / 'exports' / tag / 'public_score.json'
    if not f.exists() and tag == 'r8_base':
        f = STORED / root.parent.name / root.name / 'exports' / 'r8_base' / 'public_score.json'
    return json.loads(f.read_text()) if f.exists() else None


out = {}
for tag in ['r8_base'] + [f'A_{t}' for t in tags]:
    rows, per_clip, miss = [], {}, 0
    for root in clips:
        r, b = load(root, tag), load(root, 'r8_base')
        if r is None:
            miss += 1; continue
        per_clip[f'{root.parent.name}/{root.name}'] = dict(r['public_scores'], base=b['public_scores'])
        for n, e in r['public_errors'].items():
            be = b['public_errors'][n]
            rows.append((root.parent.name, root.name[-2:], n, e['position_cm'], e['orientation_deg'], be['position_cm'], be['orientation_deg']))
    if not rows:
        print(tag, 'no scores'); continue
    a = np.array([x[3:] for x in rows])
    pc = list(per_clip.values())
    s = dict(clips=len(pc), missing=miss, mpsr=sum(v['MP-SR'] for v in pc), spsr=sum(v['SP-SR'] for v in pc),
             pos=a[:, 0].mean(), ori=a[:, 1].mean(), mppe=np.mean([v['MPPE'] for v in pc]), auc=np.mean([v['AUC'] for v in pc]),
             base_pos=a[:, 2].mean(), base_ori=a[:, 3].mean(), base_mppe=np.mean([v['base']['MPPE'] for v in pc]), base_auc=np.mean([v['base']['AUC'] for v in pc]),
             worst_pos=a[:, 0].max(), worst_ori=a[:, 1].max(), n_pos_worse=int((a[:, 0] > a[:, 2] + 0.05).sum()), n_pos_better=int((a[:, 0] < a[:, 2] - 0.05).sum()),
             max_pos_delta=float((a[:, 0] - a[:, 2]).max()), max_ori_delta=float((a[:, 1] - a[:, 3]).max()))
    out[tag] = {k: float(v) for k, v in s.items()}
    out[tag]['per_object'] = [dict(split=x[0], episode=int(x[1]), object=x[2], pos_cm=x[3], ori_deg=x[4], base_pos_cm=x[5], base_ori_deg=x[6]) for x in rows]
    out[tag]['per_clip'] = per_clip
    if tag != 'r8_base':
        Path(f'/home/hansol/v2d/outputs/variants_20261030/A_work/compare_{tag[2:]}.json').write_text(json.dumps(out[tag], indent=1))
    print(f"{tag:12s} clips {s['clips']} (miss {miss}) MP {s['mpsr']:.0f} SP {s['spsr']:.0f} | pos {s['pos']:.3f} (base {s['base_pos']:.3f}) ori {s['ori']:.2f} ({s['base_ori']:.2f}) "
          f"MPPE {s['mppe']:.3f} ({s['base_mppe']:.3f}) AUC {s['auc']:.4f} ({s['base_auc']:.4f}) | worst {s['worst_pos']:.2f}cm/{s['worst_ori']:.1f}deg "
          f"| objs worse/better >0.05cm {s['n_pos_worse']}/{s['n_pos_better']} max delta +{s['max_pos_delta']:.2f}cm +{s['max_ori_delta']:.2f}deg")
    if verbose and tag != 'r8_base':
        for x in rows:
            print(f"    {x[0][:8]:8s} {x[1]} {x[2]:18s} {x[5]:.3f}/{x[6]:.2f} -> {x[3]:.3f}/{x[4]:.2f}")
if '--json' in sys.argv[-2:-1]:
    pass
Path('/home/hansol/v2d/outputs/variants_20261030/A_work/compare_last.json').write_text(json.dumps(out, indent=1))
