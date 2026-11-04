"""Per-package official-path outcomes (SP fail, MP fail) for ep20/ep33 against extra candidate truths, in G12 calib_extra format.
Validated against G12's out/pkg (c21 labels). usage: c1_labels.py OUT.json LABEL=SRC ...   SRC: fro | c21 | target:<dir> | bundle:<dir>"""
import sys, json, numpy as np
from tracks import *; import kit
PK = ['first', 'pf6', 'v13', 'v21C', 'v22b', 'v24', 'v25', 'v26']
def truth(src, ep):
    if src == 'fro': return bundle(FRO, ep) if ep in (14, 15, 16, 20, 25, 28, 30) else bundle(f'{R}/final_recon', ep)
    if src.startswith('target:'): return target(None, ep, f'{src[7:]}/episode_{ep:06d}')
    if src.startswith('bundle:'): return bundle(src[7:], ep)
    return target(src, ep)
def outcome(ach, ref, ep):
    p, o = kit.errors(ach, ref, ep); pm, om = p.mean(0), o.mean(0)
    sp = int(not ((pm.mean() <= 10) and (np.radians(om).mean() <= 0.5))); mp = int(not np.all((pm < 3) & (om < 30)))
    return [sp, mp], [[round(float(a), 2), round(float(b), 1)] for a, b in zip(pm, om)]
out = {}; outf = sys.argv[1]
for arg in sys.argv[2:]:
    lab, src = arg.split('=', 1)
    for ep in (20, 33):
        T = truth(src, ep); d = out.setdefault(str(ep), {}).setdefault(lab, {}); det = {}
        for p in PK:
            d[p], det[p] = outcome(package(p, ep), T, ep)
        print(ep, lab, ' '.join(f'{p}:{d[p][0]}{d[p][1]}' for p in PK), ' | v26 objs', det['v26'], ' first', det['first'], flush=True)
json.dump(out, open(outf, 'w'), indent=1)
