"""Window table per episode/object from results/h2h_epN.json (+ face/slot results). Windows: fixed 2 s (40 frames).
Per window: median depth residual (raw, mm) ours/new, median world-offset-tolerant residual ours/new, median IoU ours/new,
median track difference (centroid cm / rotation deg), and the frames where the tracks differ by > 2 cm or > 20 deg.
Automatic window winner: depth margin >= 1.5 mm (offset-tolerant) and IoU margin >= 0.05 counted as votes; ties -> '='."""
import json, sys, numpy as np
from pathlib import Path
R = Path(__file__).resolve().parents[1] / 'results'
def med(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]; return float(np.median(xs)) if xs else None
def summarize(ep, win=40):
    d = json.load(open(R / f'h2h_ep{ep}.json')); out = {}
    for k in d['names']:
        rows = [r for r in d['rows'] if k in r]; frames = [r['frame'] for r in rows]; T = max(frames) + 1; wins = []
        for a in range(0, T, win):
            rr = [r for r in rows if a <= r['frame'] < a + win]
            if not rr: continue
            g = lambda tn, key: med([r[k].get(tn, {}).get(key) for r in rr])
            w = dict(frames=[a, min(a + win, T) - 1], n=len(rr),
                     depth_mm=[g('ours', 'med_mm'), g('new', 'med_mm')], depth_off_mm=[g('ours', 'med_off_mm'), g('new', 'med_off_mm')],
                     iou=[g('ours', 'iou'), g('new', 'iou')], d_cen_cm=med([r[k]['d_cen_cm'] for r in rr]), d_rot_deg=med([r[k]['d_rot_deg'] for r in rr]))
            v = 0
            if None not in w['depth_off_mm']:
                if w['depth_off_mm'][0] < w['depth_off_mm'][1] - 1.5: v -= 1
                elif w['depth_off_mm'][1] < w['depth_off_mm'][0] - 1.5: v += 1
            if None not in w['iou']:
                if w['iou'][0] > w['iou'][1] + .05: v -= 1
                elif w['iou'][1] > w['iou'][0] + .05: v += 1
            w['auto'] = 'OURS' if v < 0 else ('NEW' if v > 0 else '=')
            wins.append(w)
        diff = [r['frame'] for r in rows if r[k]['d_cen_cm'] > 2 or r[k]['d_rot_deg'] > 20]
        # compress into ranges
        rng = []
        for f in diff:
            if rng and f - rng[-1][1] <= 6: rng[-1][1] = f
            else: rng.append([f, f])
        allg = lambda tn, key: med([r[k].get(tn, {}).get(key) for r in rows])
        out[k] = dict(windows=wins, differ_frames=rng, n_frames=len(rows),
                      overall=dict(depth_mm=[allg('ours', 'med_mm'), allg('new', 'med_mm')], depth_off_mm=[allg('ours', 'med_off_mm'), allg('new', 'med_off_mm')],
                                   iou=[allg('ours', 'iou'), allg('new', 'iou')], d_cen_cm=med([r[k]['d_cen_cm'] for r in rows]), d_rot_deg=med([r[k]['d_rot_deg'] for r in rows]),
                                   d_cen_max=max(r[k]['d_cen_cm'] for r in rows), d_rot_max=max(r[k]['d_rot_deg'] for r in rows)))
    return d, out
def fmt(x, p=1): return '-' if x is None else f'{x:.{p}f}'
if __name__ == '__main__':
    allout = {}
    for ep in [int(e) for e in sys.argv[1:]]:
        d, out = summarize(ep); allout[ep] = dict(world_offset_m=d['world_offset_m'], objects=out)
        print(f'\n### ep{ep}  world offsets cm: ' + ', '.join(f"{t}={np.round(np.array(v)*100,2).tolist()}" for t, v in d['world_offset_m'].items()))
        for k, o in out.items():
            ov = o['overall']
            print(f"{k}: overall depth {fmt(ov['depth_mm'][0])}/{fmt(ov['depth_mm'][1])} mm, off-tol {fmt(ov['depth_off_mm'][0])}/{fmt(ov['depth_off_mm'][1])}, IoU {fmt(ov['iou'][0],2)}/{fmt(ov['iou'][1],2)}, diff med {fmt(ov['d_cen_cm'])} cm/{fmt(ov['d_rot_deg'],0)} deg (max {fmt(ov['d_cen_max'])}/{fmt(ov['d_rot_max'],0)}); differ >2cm/20deg: {o['differ_frames']}")
            print('| frames | depth mm ours/new | off-tol mm | IoU ours/new | diff cm/deg | auto |')
            for w in o['windows']:
                print(f"| {w['frames'][0]}-{w['frames'][1]} | {fmt(w['depth_mm'][0])}/{fmt(w['depth_mm'][1])} | {fmt(w['depth_off_mm'][0])}/{fmt(w['depth_off_mm'][1])} | {fmt(w['iou'][0],2)}/{fmt(w['iou'][1],2)} | {fmt(w['d_cen_cm'])}/{fmt(w['d_rot_deg'],0)} | {w['auto']} |")
    json.dump(allout, open(R / 'summary.json', 'w'), indent=1)
