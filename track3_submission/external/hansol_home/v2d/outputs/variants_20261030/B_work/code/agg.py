"""Variant B: per object type x family table from public_score.json files in our mirror (GT entered only the scorer)."""
import json, glob, os, sys
from collections import defaultdict
W = '/home/hansol/v2d/outputs/variants_20261030/B_work'
PATTERN = sys.argv[1] if len(sys.argv) > 1 else 'fam_*'
rows = defaultdict(dict)   # (split, ep, obj) -> fam -> (pos, ori)
for f in glob.glob(f'{W}/canonical/*/episode_*/exports/{PATTERN}/public_score.json'):
    p = f.split('/'); split, ep, fam = p[-5], int(p[-4][-6:]), p[-2]
    r = json.load(open(f))
    for o, e in r['public_errors'].items(): rows[(split, ep, o)][fam] = (e['position_cm'], e['orientation_deg'])
fams = sorted({f for v in rows.values() for f in v})
typ = defaultdict(list)
for k, v in rows.items(): typ[k[2]].append((k, v))
out = {}
print('per object (pos cm / ori deg); * = MP fail (>=3 cm or >=30 deg)')
for (split, ep, o), v in sorted(rows.items(), key=lambda x: (x[0][2], x[0][0], x[0][1])):
    print(f'{o:18s} {split[:7]:7s} {ep:3d} ' + ' '.join(f'{fm[4:]}={v[fm][0]:.2f}/{v[fm][1]:.1f}{"*" if v[fm][0] >= 3 or v[fm][1] >= 30 else ""}' for fm in fams if fm in v))
print('\nper type: family n pos_mean ori_mean pos_max ori_max mp_fail  | paired vs base on the same objects: dpos dori wins')
for o, items in sorted(typ.items()):
    out[o] = {}
    for fm in fams:
        xs = [(k, v[fm]) for k, v in items if fm in v]
        if not xs: continue
        pm = sum(x[1][0] for x in xs) / len(xs); om = sum(x[1][1] for x in xs) / len(xs)
        fail = sum(x[1][0] >= 3 or x[1][1] >= 30 for x in xs)
        pb = [(v[fm], v['fam_base']) for k, v in items if fm in v and 'fam_base' in v]
        dp = sum(a[0] - b[0] for a, b in pb) / len(pb); do = sum(a[1] - b[1] for a, b in pb) / len(pb)
        wins = sum(a[0] < b[0] and a[1] < b[1] + 1e-9 for a, b in pb)
        out[o][fm] = dict(n=len(xs), pos_mean=pm, ori_mean=om, pos_max=max(x[1][0] for x in xs), ori_max=max(x[1][1] for x in xs), mp_fail=fail, dpos_vs_base=dp, dori_vs_base=do, wins=wins)
        print(f'{o:18s} {fm[4:]:20s} {len(xs):2d} {pm:6.2f} {om:6.2f} {out[o][fm]["pos_max"]:6.2f} {out[o][fm]["ori_max"]:7.1f} {fail:2d} | {dp:+6.2f} {do:+7.2f} {wins}/{len(pb)}')
json.dump(dict(per_object={f'{k[0]}/{k[1]}/{k[2]}': v for k, v in rows.items()}, per_type=out), open(f'{W}/per_type_{PATTERN.replace("*", "X")}.json', 'w'), indent=1)
