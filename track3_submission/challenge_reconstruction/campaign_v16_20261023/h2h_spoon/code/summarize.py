"""Per-episode, per-object, per-window (1 s) head-to-head table + automatic window votes.
Votes per window (only where the tracks differ > 1 cm or > 10 deg; else TIE):
  markers (spoon only, canonical balls): inlier(<15 px) fraction difference >= 0.25 with >= 4 detections -> vote (weight 2, independent)
  depth: in10 reported only (NOT voted): NEW is ~1 cm off this stereo even at rest where markers/IoU agree -> stereo-pipeline bias toward OURS
  free space: fs difference >= 0.10 -> vote (weight 1)
  silhouette: IoU difference >= 0.05 -> vote (weight 1; ours fit these masks, NEW refined to an RGB mask)
usage: summarize.py OUT.json EP..."""
import sys, json, numpy as np
D = '/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/h2h_spoon'
out = sys.argv[1]; eps = [int(x) for x in sys.argv[2:]]; canon = json.load(open(D + '/markers/canon.json'))['eps']
nan = float('nan'); res = {}
def med(a): a = np.array([x for x in a if x is not None and np.isfinite(x)]); return float(np.median(a)) if len(a) else nan
for ep in eps:
    z = json.load(open(f'{D}/evidence/ev_{ep}.json')); rows = z['rows']; N = rows[-1]['frame'] + 1; res[ep] = {}
    mk = canon.get(str(ep))
    for o in z['names']:
        sym = o in ('blue_cup',)
        W = []
        for s0 in range(0, N, 20):
            rr = [r for r in rows if s0 <= r['frame'] < s0 + 20]
            g = lambda tr, k: med([r[o][tr].get(k) for r in rr])
            w = dict(f0=s0, f1=min(s0 + 19, N - 1), diff_cm=med([float(np.linalg.norm(r[o]['off_body_cm'])) if sym else r[o]['diff_cm'] for r in rr]), diff_deg=med([r[o]['axis_deg' if sym else 'diff_deg'] for r in rr]),
                     along_cm=med([r[o]['off_body_cm'][0] for r in rr]))
            for tr in ('ours', 'new'):
                for k in ('dm', 'in10', 'fs', 'iou'): w[f'{k}_{tr}'] = g(tr, k)
            if o == 'wooden_spoon' and mk:
                F = np.array(mk['frames']); sel = (F >= s0) & (F < s0 + 20); w['mk_n'] = int(sel.sum())
                for tr in ('ours', 'new'):
                    a = np.array(mk[tr])[sel]; t = np.array(mk['twin_' + tr])[sel]
                    w[f'mk15_{tr}'] = float(np.mean(a < 15)) if sel.sum() else nan; w[f'mkmed_{tr}'] = float(np.median(a)) if sel.sum() else nan
                    w[f'mktwin15_{tr}'] = float(np.mean(t < 15)) if sel.sum() else nan
            v = 0.0; why = []
            if (w['diff_cm'] > 1.0 or w['diff_deg'] > 10):
                if w.get('mk_n', 0) >= 4 and abs(w['mk15_ours'] - w['mk15_new']) >= 0.25: v += 2 * np.sign(w['mk15_ours'] - w['mk15_new']); why.append('mk')
                if np.isfinite(w['in10_ours'] - w['in10_new']) and abs(w['in10_ours'] - w['in10_new']) >= 0.10: why.append('(depth)')  # reported, not voted: biased to OURS (see README)
                if np.isfinite(w['fs_ours'] - w['fs_new']) and abs(w['fs_ours'] - w['fs_new']) >= 0.10: v += np.sign(w['fs_new'] - w['fs_ours']); why.append('fs')
                if np.isfinite(w['iou_ours'] - w['iou_new']) and abs(w['iou_ours'] - w['iou_new']) >= 0.05: v += np.sign(w['iou_ours'] - w['iou_new']); why.append('iou')
                w['vote'] = 'OURS' if v >= 2 else ('NEW' if v <= -2 else ('ours?' if v > 0 else ('new?' if v < 0 else 'undecided')))
            else: w['vote'] = 'TIE'
            w['score'] = float(v); w['why'] = why; W.append(w)
        res[ep][o] = W
        print(f'\n== ep{ep} {o}')
        print(' frames  | diff cm/deg along | in10 o/n  | fs o/n    | IoU o/n   | mk n  inl15 o/n  twin o/n | vote')
        for w in W:
            mks = f"{w.get('mk_n', 0):3d}  {w.get('mk15_ours', nan):.2f}/{w.get('mk15_new', nan):.2f}  {w.get('mktwin15_ours', nan):.2f}/{w.get('mktwin15_new', nan):.2f}" if 'mk_n' in w else ' ' * 26
            print(f"{w['f0']:3d}-{w['f1']:3d} | {w['diff_cm']:4.1f}/{w['diff_deg']:5.1f} {w['along_cm']:+5.1f} | {w['in10_ours']:.2f}/{w['in10_new']:.2f} | {w['fs_ours']:.2f}/{w['fs_new']:.2f} | {w['iou_ours']:.2f}/{w['iou_new']:.2f} | {mks} | {w['vote']} {w['score']:+.0f} {','.join(w['why'])}")
json.dump(res, open(out, 'w'), indent=0)
