"""Compare public_score.json of export tags vs g15_camrt over tuning / held-out (scoring outputs only; clips without the
tag fall back to g15_camrt, i.e. unchanged). Usage: cmp17.py TAG [TAG ...] [-v]"""
import json, sys, numpy as np
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/streamG17_place/code')
from g17lib import eproot, TUNE, HELD
tags = [t for t in sys.argv[1:] if t != '-v']; verbose = '-v' in sys.argv
def load(e, t):
    f = eproot('public', e) / 'exports' / t / 'public_score.json'
    if not f.exists(): f = eproot('public', e) / 'exports' / 'g15_camrt' / 'public_score.json'; fb = True
    else: fb = False
    return json.loads(f.read_text()), fb
for nm, eps in [('tune', TUNE), ('held', HELD)]:
    for t in ['g15_camrt'] + tags:
        P, O, mp, auc, mppe, miss = [], [], [], [], [], 0
        for e in eps:
            r, fb = load(e, t)
            for n, v in r['public_errors'].items():
                P.append(v['position_cm']); O.append(v['orientation_deg'])
                if verbose and t != 'g15_camrt' and not fb:
                    b = load(e, 'g15_camrt')[0]['public_errors'][n]
                    print(f"   {nm} {e:3d} {n:18s} {b['position_cm']:.2f}/{b['orientation_deg']:.1f} -> {v['position_cm']:.2f}/{v['orientation_deg']:.1f}")
            sc = r['public_scores']; mp.append(sc['MP-SR']); auc.append(sc.get('AUC', np.nan)); mppe.append(sc.get('MPPE', np.nan))
        print(f"{nm:5s} {t:16s} MP {int(sum(mp))}/{len(mp)} pos {np.mean(P):.3f} (max {np.max(P):.2f}) ori {np.mean(O):.2f} (max {np.max(O):.1f}) MPPE {np.mean(mppe):.4f} AUC {np.mean(auc):.4f}")
