"""Summarise the stereo-depth time-shift scan: per object and segment, median |dz| on moving frames for each shift s.
usage: shiftscan.py BUNDLE EP SPLIT EVJSON_PATTERN(with {s}) S1 S2 ..."""
import sys, json, numpy as np
B, ep, k, pat = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]; S = sys.argv[5:]
W = np.load(f'{B}/episode_{ep:06d}/object_poses.npz')['world_from_object']
ev = {s: json.load(open(pat.format(s=s)))['per_frame'] for s in S}
arr = lambda r, key: np.array([np.nan if x is None else x for x in r[key]], float)
names = list(ev[S[0]])
out = {}
for o, n in enumerate(names):
    sp = np.r_[0, np.linalg.norm(np.diff(W[:, o, :3, 3], axis=0), axis=1) * 100]
    for lab, sel in ((f'f0-{k - 1}', np.arange(len(sp)) < k), (f'f{k}-end', np.arange(len(sp)) >= k)):
        m = sel & (sp > 0.5)
        A = np.array([arr(ev[s][n], 'adz')[m] for s in S]); ok = np.isfinite(A).all(0)
        I = np.array([arr(ev[s][n], 'iou')[m] for s in S])[:, ok]
        med = np.median(A[:, ok], axis=1); best = S[int(np.argmin(med))]
        # per-frame argmin shift (vote)
        votes = np.bincount(np.argmin(A[:, ok], axis=0), minlength=len(S)) / max(1, ok.sum())
        print(f'{n:14s} {lab:9s} n{ok.sum():3d} median|dz| mm ' + ' '.join(f's{s}:{v:5.2f}' for s, v in zip(S, med)) + f'  -> best s={best}')
        print(f'{"":24s} per-frame argmin share  ' + ' '.join(f's{s}:{v:4.0%}' for s, v in zip(S, votes)) + '   IoU ' + ' '.join(f'{np.median(x):.3f}' for x in I))
        out[f'{n}|{lab}'] = dict(n=int(ok.sum()), median_abs_dz_mm=dict(zip(S, np.round(med, 3).tolist())), best=best, argmin_share=dict(zip(S, np.round(votes, 3).tolist())))
json.dump(out, open(pat.format(s='scan_summary').replace('.json', '') + '.json', 'w'), indent=1)
