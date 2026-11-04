import json, glob, numpy as np
for e in [20,33,3,1,19,35,36,5,6,8,10,14,15,16,25,28,30,32,37,38]:
    try: d = json.load(open(f'../out/audit/ep{e}.json'))
    except Exception: print(e, 'missing'); continue
    for n, o in d['objects'].items():
        q = o['qc']; tm = ' '.join(f"{k[:3]} {v['pos']:.1f}/{v['rot']:.0f}/{v['frac_bad']:.2f}" for k, v in o.get('tm', {}).items())
        W = o['windows']; tb = [w for w in W if (w['iou_trk'] > w['iou_ours'] + 0.03) or (w['dep_trk'] is not None and w['dep_ours'] is not None and w['dep_trk'] < w['dep_ours'] - 0.3 and w['iou_trk'] >= w['iou_ours'] - 0.01)]
        nb = sum(w['f1'] - w['f0'] + 1 for w in W)
        tw = ' '.join(f"{k[-1]}:{v['twin_better_iou_frames']}/{v['ours_better_iou_frames']}({v['iou_ours']:.2f}v{v['iou_twin']:.2f})" for k, v in o['twins'].items())
        print(f"{e:3d} {n:16s} IoU {q['iou_mean'] or 0:.2f} p10 {q['iou_p10'] or 0:.2f} dep {q['dep_med'] or 0:.2f}/{q['dep_p90'] or 0:.2f} lowIoU {len(q['low_iou_frames'])} bigdep {len(q['big_dep_frames'])} | {tm} | win {len(W)} ({nb} fr) trk-better {len(tb)} | twins {tw}")
        for w in tb: print(f"      trk-better {w['tracker'][:4]} f{w['f0']}-{w['f1']} {w['pos_mean']:.1f}cm/{w['rot_mean']:.0f}deg IoU {w['iou_ours']:.2f}->{w['iou_trk']:.2f} dep {w['dep_ours']}->{w['dep_trk']}")
        if q['low_iou_frames']: print('      lowIoU', q['low_iou_frames'][:30])
