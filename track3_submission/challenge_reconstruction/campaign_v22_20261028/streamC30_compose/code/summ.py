"""Print score_any results for labels matching a pattern next to v29's scores vs target_final_v14_g17.
usage: summ.py PATTERN   (glob over campaign_v22_20261028/scores/cand/)"""
import sys, json, glob, os
C = '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028'
v29 = json.load(open('/data1/SRL_V2D/challenge_submissions_SRL/track3_v29_20261030/all_20_episodes/target_final_v14_g17_scores.json'))
for d in sorted(glob.glob(f'{C}/scores/cand/{sys.argv[1]}')):
    f = d + '/score.json'
    if not os.path.exists(f): print(os.path.basename(d), 'MISSING'); continue
    s = json.load(open(f)); e = list(s)[0]; v = s[e]; b = v29[e]
    o = ' '.join(f'{n} {x[0]:.2f}/{x[1]:.1f}' for n, x in v['objects'].items())
    ob = ' '.join(f'{x[0]:.2f}/{x[1]:.1f}' for n, x in b['objects'].items())
    sc = v['scores']
    print(f"{os.path.basename(d):34s} MP {sc['MP-SR']:.0f} SP {sc.get('SP-SR', float('nan')):.0f} MPPE {sc['MPPE']:.3f} AUC {sc['AUC']:.4f} | {o} || v29@v14 MPPE {b['scores']['MPPE']:.3f} AUC {b['scores']['AUC']:.4f} {ob}")
