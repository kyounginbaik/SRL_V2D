"""Summarise rdepth_epN.json: per object and 2 s window, median |dz| mm and 'agree' fraction, ours/new."""
import json, sys, numpy as np
from pathlib import Path
R = Path(__file__).resolve().parents[1] / 'results'; allout = {}
for ep in sys.argv[1:]:
    d = json.load(open(R / f'rdepth_ep{ep}.json')); allout[ep] = {}
    for k in d['names']:
        rows = [r for r in d['rows'] if k in r]; T = max(r['frame'] for r in rows) + 1; ws = []
        for a in range(0, T, 40):
            rr = [r[k] for r in rows if a <= r['frame'] < a + 40]
            g = lambda tn, key: (lambda x: float(np.median(x)) if x else None)([x[tn][key] for x in rr if x.get(tn) and key in x[tn]])
            ws.append(dict(frames=[a, min(a + 40, T) - 1], med_mm=[g('ours', 'med_mm'), g('new', 'med_mm')], agree=[g('ours', 'agree'), g('new', 'agree')]))
        g = lambda tn, key: (lambda x: float(np.median(x)) if x else None)([r[k][tn][key] for r in rows if r[k].get(tn) and key in r[k][tn]])
        allout[ep][k] = dict(overall=dict(med_mm=[g('ours', 'med_mm'), g('new', 'med_mm')], agree=[g('ours', 'agree'), g('new', 'agree')]), windows=ws)
        f = lambda x: '-' if x is None else f'{x:.2f}'
        print(f"ep{ep} {k}: overall |dz| {f(allout[ep][k]['overall']['med_mm'][0])}/{f(allout[ep][k]['overall']['med_mm'][1])} mm agree {f(allout[ep][k]['overall']['agree'][0])}/{f(allout[ep][k]['overall']['agree'][1])}  windows agree: " + ' '.join(f"{w['frames'][0]}:{f(w['agree'][0])}/{f(w['agree'][1])}" for w in ws))
json.dump(allout, open(R / 'rdepth_summary.json', 'w'), indent=1)
