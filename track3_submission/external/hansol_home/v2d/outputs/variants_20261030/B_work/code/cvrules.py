"""Variant B: 2-fold check of rule selection. Rebuild the per-type table from ONE split's per-object scores, derive rules with
the same type_rules(), and compare with the all-20 rules. GT entered only the scorer that produced these numbers."""
import json, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(__file__)); import variants as V
W = '/home/hansol/v2d/outputs/variants_20261030/B_work'
po = json.load(open(f'{W}/per_type_fam_X.json'))['per_object']
def table(sel):
    typ = {}
    for k, v in po.items():
        split, ep, o = k.split('/')
        if not sel(split) or 'fam_base' not in v: continue
        typ.setdefault(o, []).append(v)
    out = {}
    for o, items in typ.items():
        out[o] = {}
        for fm in {f for v in items for f in v}:
            xs = [v[fm] for v in items if fm in v]; pb = [(v[fm], v['fam_base']) for v in items if fm in v]
            out[o][fm] = dict(n=len(xs), mp_fail=sum(x[0] >= 3 or x[1] >= 30 for x in xs),
                              dpos_vs_base=float(np.mean([a[0] - b[0] for a, b in pb])), dori_vs_base=float(np.mean([a[1] - b[1] for a, b in pb])))
    return out
res = {}
for name, sel in [('all', lambda s: True), ('public_only', lambda s: s == 'public'), ('heldout_only', lambda s: s == 'public_heldout')]:
    b1, good = V.type_rules(table(sel)); res[name] = dict(B1=b1, B2_good=good); print(name, 'B1', b1); print(name, 'B2', good)
json.dump(res, open(f'{W}/cv_rules.json', 'w'), indent=1)
