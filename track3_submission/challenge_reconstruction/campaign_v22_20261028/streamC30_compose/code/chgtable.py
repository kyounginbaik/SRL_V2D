"""Markdown table of the compose_vK changes (from out/changes_vK_fix.json). usage: chgtable.py K"""
import json, sys
k = sys.argv[1]; r = json.load(open(f'/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamC30_compose/out/changes_v{k}_fix.json'))
def w(ws): 
    s = ','.join(f'{a}-{b}' if a != b else f'{a}' for a, b in ws); return s if len(s) < 70 else s[:67] + '...'
print(f'| ep | object | kind | frames changed (vs v8_g10b, RGB clock) | n | max vs v14 cm/deg | mean vs v14 cm/deg | frames >1cm/5deg |')
print('|---|---|---|---|---|---|---|---|')
for e, er in r['episodes'].items():
    for n, o in er['objects'].items():
        if not o['n_changed'] and not o['whole_track']: continue
        print(f"| {e} | {n} | {'whole track' if o['whole_track'] else 'windows/smoothing'} | {w(o['windows'])} | {o['n_changed']} | "
              f"{o.get('vs_v14_cm_max',0):.2f}/{o.get('vs_v14_deg_max',0):.1f} | {o.get('vs_v14_cm_mean',0):.2f}/{o.get('vs_v14_deg_mean',0):.1f} | {o.get('vs_v14_n_gt1cm_or_5deg',0)} |")
