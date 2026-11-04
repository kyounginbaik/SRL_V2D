"""compare.py variant for G14: clips whose object_poses are bit-identical to r8_base take r8_base's score (deterministic scorer);
every other clip must have its own public_score.json. GT used only by the scorer."""
import json, sys, numpy as np
from pathlib import Path
W = Path('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/canonical')
clips = sorted(list((W/'public').glob('episode_*')) + list((W/'public_heldout').glob('episode_*')))
out = {}
for tag in sys.argv[1:]:
    rows = []; clipr = []; miss = 0; nchg = 0
    for root in clips:
        b = json.loads((root/'exports/r8_base/public_score.json').read_text()); d = root/'exports'/tag
        same = np.abs(np.load(d/'object_poses.npz')['world_from_object'] - np.load(root/'exports/r8_base/object_poses.npz')['world_from_object']).max() < 1e-9
        if same: r = b
        elif (d/'public_score.json').exists(): r = json.loads((d/'public_score.json').read_text()); nchg += 1
        else: miss += 1; continue
        clipr.append((r['public_scores']['MP-SR'], r['public_scores']['MPPE'], r['public_scores']['AUC'], b['public_scores']['MPPE'], b['public_scores']['AUC']))
        for n, e in r['public_errors'].items():
            be = b['public_errors'][n]; rows.append((root.name[-2:], n, e['position_cm'], e['orientation_deg'], be['position_cm'], be['orientation_deg']))
    c = np.array(clipr); a = np.array([x[2:] for x in rows])
    print(f"{tag:10s} changed clips {nchg} missing {miss} | MP {c[:,0].sum():.0f}/{len(c)} | pos {a[:,0].mean():.3f} (base {a[:,2].mean():.3f}) ori {a[:,1].mean():.2f} (base {a[:,3].mean():.2f}) | MPPE {c[:,1].mean():.4f} (base {c[:,3].mean():.4f}) AUC {c[:,2].mean():.4f} (base {c[:,4].mean():.4f})")
    for x in rows:
        if abs(x[2]-x[4]) > .005 or abs(x[3]-x[5]) > .05: print(f"    {x[0]} {x[1]:18s} {x[4]:.2f}/{x[5]:.1f} -> {x[2]:.2f}/{x[3]:.1f}")
    out[tag] = dict(changed=nchg, missing=miss, mp=float(c[:,0].sum()), n=len(c), mppe=float(c[:,1].mean()), auc=float(c[:,2].mean()), pos=float(a[:,0].mean()), ori=float(a[:,1].mean()))
json.dump(out, open('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG14/out/public_compare.json', 'w'), indent=1)
