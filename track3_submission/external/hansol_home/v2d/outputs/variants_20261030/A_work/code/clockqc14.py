"""Clock-consistent eval QC of the v14-based variants from the guard reports (IoU / ball on the RGB clock = track shifted by
-delta; |depth residual| on the stereo clock = track as stored), per object and in total, plus large-step counts.
usage: clockqc14.py TAG   (A_work/eval14/TAG) -> A_work/clockqc14_TAG.json"""
import sys, json
from pathlib import Path
import numpy as np
W = Path('/home/hansol/v2d/outputs/variants_20261030/A_work')
tag = sys.argv[1]
res, T = {}, dict(ib=[], ia=[], zb=[], za=[], bb=[], ba=[], drop=0)
arr = lambda x: np.array([np.nan if v is None else v for v in x], float)
for f in sorted((W / 'eval14' / tag).glob('episode_*/guard_report.json')):
    rep = json.loads(f.read_text()); ep = f.parent.name; res[ep] = {}
    for n, o in rep['objects'].items():
        pf = o['per_frame']; ib, ia = map(arr, pf['iou_rgbclock']); zb, za = map(arr, pf['absz_stereoclock']); bb, ba = map(arr, pf['ball_rgbclock'])
        nm = lambda x: float(np.nanmean(x)) if np.isfinite(x).any() else None
        md = lambda x: float(np.nanmedian(x)) if np.isfinite(x).any() else None
        p9 = lambda x: float(np.nanpercentile(x, 90)) if np.isfinite(x).any() else None
        r = dict(iou=[nm(ib), nm(ia)], absz=[nm(zb), nm(za)], iou_drop_gt0p1=int(np.nansum(ia - ib < -.1)), ball_median=[md(bb), md(ba)], ball_p90=[p9(bb), p9(ba)],
                 rot_gt30=[o['before']['rot_gt30'], o['after']['rot_gt30']], rot_gt15=[o['before']['rot_gt15'], o['after']['rot_gt15']],
                 pos_gt3=[o['before']['pos_gt3'], o['after']['pos_gt3']], racc_gt20=[o['before']['racc_gt20'], o['after']['racc_gt20']],
                 change_rot_mean=o['change_rot_mean'], change_pos_mean=o['change_pos_mean'], kept_jumps=o['cuts'], final_bad=o['final_bad'])
        res[ep][n] = r
        T['ib'] += list(ib); T['ia'] += list(ia); T['zb'] += list(zb); T['za'] += list(za); T['bb'] += list(bb); T['ba'] += list(ba); T['drop'] += r['iou_drop_gt0p1']
        f2 = lambda x: '-' if x is None else f'{x:.2f}'
        print(f"{ep} {n:18s} IoU {r['iou'][0]:.3f}->{r['iou'][1]:.3f} drop>.1 {r['iou_drop_gt0p1']} |z| {r['absz'][0]:.2f}->{r['absz'][1]:.2f} ball med {f2(r['ball_median'][0])}->{f2(r['ball_median'][1])} p90 {f2(r['ball_p90'][0])}->{f2(r['ball_p90'][1])} "
              f">30 {r['rot_gt30']} >15 {r['rot_gt15']} pos>3 {r['pos_gt3']} racc>20 {r['racc_gt20']} final_bad {r['final_bad']}")
m = lambda k: float(np.nanmean(np.array(T[k], float)))
tot = lambda k, i: sum(r[k][i] for e in res.values() for r in e.values())
S = dict(iou_before=m('ib'), iou_after=m('ia'), absz_before=m('zb'), absz_after=m('za'), ball_before=m('bb'), ball_after=m('ba'), iou_drop_gt0p1=T['drop'],
         rot_gt30=[tot('rot_gt30', 0), tot('rot_gt30', 1)], rot_gt15=[tot('rot_gt15', 0), tot('rot_gt15', 1)], pos_gt3=[tot('pos_gt3', 0), tot('pos_gt3', 1)], racc_gt20=[tot('racc_gt20', 0), tot('racc_gt20', 1)])
print('TOTAL', S)
(W / f'clockqc14_{tag}.json').write_text(json.dumps(dict(summary=S, objects=res), indent=1))
