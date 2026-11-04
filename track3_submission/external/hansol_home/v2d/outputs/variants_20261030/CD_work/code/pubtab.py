"""Tabulate stored unchanged-scorer public_score.json of two exports over public 10 + held-out 10.
usage: pubtab.py TAG_A TAG_B"""
import sys, json, glob, os
CAN = '/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/canonical'
A, B = sys.argv[1], sys.argv[2]
tot = {A: [], B: []}
for split in ('public', 'public_heldout'):
    for ep in sorted(glob.glob(f'{CAN}/{split}/episode_*')):
        row = []
        for t in (A, B):
            f = f'{ep}/exports/{t}/public_score.json'
            if not os.path.exists(f): row.append(None); continue
            d = json.load(open(f)); row.append(d)
        if None in row: print(split, ep[-2:], 'missing', [r is None for r in row]); continue
        def s(d):
            sc = d.get('public_scores', d.get('scores', {})); er = d.get('public_errors', d.get('errors', {}))
            return sc, er
        (sa, ea), (sb, eb) = s(row[0]), s(row[1])
        tot[A].append(sa); tot[B].append(sb)
        print(split[:8], ep[-2:], 'MP', sa.get('MP-SR'), '->', sb.get('MP-SR'), '| MPPE', round(sa.get('MPPE', float('nan')), 3), '->', round(sb.get('MPPE', float('nan')), 3),
              '| AUC', round(sa.get('AUC', float('nan')), 4), '->', round(sb.get('AUC', float('nan')), 4))
        for o in ea:
            if isinstance(ea[o], dict): print('      ', o, {k: round(v, 2) for k, v in ea[o].items() if isinstance(v, float)}, '->', {k: round(v, 2) for k, v in eb.get(o, {}).items() if isinstance(v, float)})
if tot[A] and not isinstance(tot[A][0], dict): sys.exit()
import numpy as np
for t in (A, B):
    L = tot[t]
    print(t, 'n', len(L), 'MP pass', sum(x.get('MP-SR', 0) >= 1 for x in L), 'MPPE mean', round(np.mean([x['MPPE'] for x in L]), 4), 'AUC mean', round(np.mean([x['AUC'] for x in L]), 4))
