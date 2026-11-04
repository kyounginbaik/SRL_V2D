"""Compare public_score.json of two export tags over tuning / held-out (scoring outputs only)."""
import json, sys, numpy as np, kitx
a, b = sys.argv[1], sys.argv[2]
for nm, eps in [('tune', kitx.TUNE), ('held', kitx.HELD), ('all', kitx.TUNE + kitx.HELD)]:
    S = {t: dict(P=[], O=[], mp=[], auc=[], mppe=[]) for t in (a, b)}
    for e in eps:
        for t in (a, b):
            f = kitx.root(e) / 'exports' / t / 'public_score.json'
            if not f.exists(): f = kitx.root(e) / 'exports' / 'r8_base' / 'public_score.json'
            r = json.loads(f.read_text())
            for n, v in r['public_errors'].items():
                S[t]['P'].append(v['position_cm']); S[t]['O'].append(v['orientation_deg'])
                if nm != 'all' and len(sys.argv) > 3 and t == b: print(f"  {e:3d} {n:18s} {a} {json.loads((kitx.root(e)/'exports'/a/'public_score.json').read_text())['public_errors'][n]['position_cm']:.2f}/{json.loads((kitx.root(e)/'exports'/a/'public_score.json').read_text())['public_errors'][n]['orientation_deg']:.1f} -> {v['position_cm']:.2f}/{v['orientation_deg']:.1f}")
            sc = r['public_scores']; S[t]['mp'].append(sc['MP-SR']); S[t]['auc'].append(sc.get('AUC', np.nan)); S[t]['mppe'].append(sc.get('MPPE', np.nan))
    for t in (a, b):
        s = S[t]; print(f"{nm:5s} {t:22s} MP {int(sum(s['mp']))}/{len(s['mp'])} pos {np.mean(s['P']):.3f} (max {np.max(s['P']):.2f}) ori {np.mean(s['O']):.2f} (max {np.max(s['O']):.1f}) MPPE {np.mean(s['mppe']):.3f} AUC {np.mean(s['auc']):.4f}")
