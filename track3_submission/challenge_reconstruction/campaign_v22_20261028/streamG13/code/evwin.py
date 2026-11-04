"""Windowed summary of ev.py outputs. usage: evwin.py OUT1.json OUT2.json ... [--win a-b,c-d]"""
import sys, json, numpy as np
fs = [a for a in sys.argv[1:] if not a.startswith('--')]; W = [(0, 1), (0, 10), (10, 50), (50, 80), (80, 110), (110, 150), (150, 202)]
for a in sys.argv[1:]:
    if a.startswith('--win='): W = [tuple(int(x) for x in w.split('-')) for w in a[6:].split(',')]
D = {f.split('/')[-1][:-5]: json.load(open(f)) for f in fs}
def m(x):
    x = [v for v in x if v is not None]; return np.mean(x) if x else np.nan
names = list(D.values())[0]['names']
for n in names:
    print('==', n, ' window: IoU | dz(signed mm) | |dz| | centroid dx,dy px(full res)')
    for a, b in W:
        s = f'  f{a:3d}-{b:3d} '
        for k, d in D.items():
            r = d['per_frame'][n]; s += f'| {k[:14]:14s} {m(r["iou"][a:b]):.3f} {m(r["dz"][a:b]):+6.1f} {m(r["adz"][a:b]):5.1f} {m(r["dcx"][a:b]):+5.0f},{m(r["dcy"][a:b]):+5.0f} '
        print(s)
