"""Clock-consistent eval QC (GT-free) of a guarded variant built on another v14-clock base, against that base and against v14_g17.
IoU / ball on the RGB clock (track shifted by -delta), |depth residual| on the stereo clock (as stored), large-step counts.
  vs base (e.g. B2): the variant's own guard reports (before = base, after = variant)
  vs v14_g17:        before arrays from the A14 guard reports on v14 (REF tag, same evidence code), after = variant
usage: clockqc_b2a.py TAG REF   e.g. clockqc_b2a.py B2A_g8 g8 -> A_work/clockqc_b2a_TAG.json"""
import sys, json
from pathlib import Path
import numpy as np
W = Path('/home/hansol/v2d/outputs/variants_20261030/A_work')
tag, ref = sys.argv[1], sys.argv[2]
arr = lambda x: np.array([np.nan if v is None else v for v in x], float)
nm = lambda x: float(np.nanmean(x)) if np.isfinite(x).any() else None
md = lambda x: float(np.nanmedian(x)) if np.isfinite(x).any() else None
KEYS = ['rot_gt30', 'rot_gt15', 'pos_gt3', 'racc_gt20']
res, T = {}, {c: dict(ib=[], ia=[], zb=[], za=[], bb=[], ba=[], drop=0, steps={k: [0, 0] for k in KEYS}) for c in ['vs_base', 'vs_v14']}
for f in sorted((W / 'eval14' / tag).glob('episode_*/guard_report.json')):
    ep = f.parent.name; rep = json.loads(f.read_text()); v14 = json.loads((W / 'eval14' / ref / ep / 'guard_report.json').read_text())
    res[ep] = {}
    for n, o in rep['objects'].items():
        pf, pv = o['per_frame'], v14['objects'][n]['per_frame']
        after = [arr(pf['iou_rgbclock'][1]), arr(pf['absz_stereoclock'][1]), arr(pf['ball_rgbclock'][1])]
        r = {}
        for c, (pb, sb) in dict(vs_base=(pf, o['before']), vs_v14=(pv, v14['objects'][n]['before'])).items():
            ib, zb, bb = arr(pb['iou_rgbclock'][0]), arr(pb['absz_stereoclock'][0]), arr(pb['ball_rgbclock'][0])
            ia, za, ba = after
            r[c] = dict(iou=[nm(ib), nm(ia)], absz=[nm(zb), nm(za)], ball_median=[md(bb), md(ba)], iou_drop_gt0p1=int(np.nansum(ia - ib < -.1)),
                        **{k: [sb[k], o['after'][k]] for k in KEYS})
            t = T[c]; t['ib'] += list(ib); t['ia'] += list(ia); t['zb'] += list(zb); t['za'] += list(za); t['bb'] += list(bb); t['ba'] += list(ba)
            t['drop'] += r[c]['iou_drop_gt0p1']
            for k in KEYS:
                t['steps'][k][0] += sb[k]; t['steps'][k][1] += o['after'][k]
        r['change_rot_mean'] = o['change_rot_mean']; r['change_pos_mean'] = o['change_pos_mean']; r['kept_jumps'] = o['cuts']; r['final_bad'] = o['final_bad']
        res[ep][n] = r
        f2 = lambda x: '-' if x is None else f'{x:.3f}'
        print(ep, f'{n:18s}', ' | '.join(f"{c}: IoU {f2(r[c]['iou'][0])}->{f2(r[c]['iou'][1])} |z| {f2(r[c]['absz'][0])}->{f2(r[c]['absz'][1])} "
                                         f"ball {f2(r[c]['ball_median'][0])}->{f2(r[c]['ball_median'][1])} drop {r[c]['iou_drop_gt0p1']} >15 {r[c]['rot_gt15']}" for c in T))
m = lambda t, k: float(np.nanmean(np.array(t[k], float)))
S = {c: dict(iou=[m(t, 'ib'), m(t, 'ia')], absz=[m(t, 'zb'), m(t, 'za')], ball_mean=[m(t, 'bb'), m(t, 'ba')], iou_drop_gt0p1=t['drop'], **t['steps']) for c, t in T.items()}
for c, s in S.items():
    print('TOTAL', c, s)
(W / f'clockqc_b2a_{tag}.json').write_text(json.dumps(dict(tag=tag, v14_ref=ref, summary=S, objects=res), indent=1))
