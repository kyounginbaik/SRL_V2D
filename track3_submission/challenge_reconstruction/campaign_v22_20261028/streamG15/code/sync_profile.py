"""GT-free per-frame sync delta(t) from the fine profiles: piecewise-constant fit with integer-frame steps (frame drops/dups).
Writes out/sync_delta_t.json {split_ep: [delta per frame]} and a summary."""
import json, glob, numpy as np
out, summ = {}, {}
for f in sorted(glob.glob('../out/syncfine_*.json')):
    r = json.load(open(f)); P = np.array(r['prof']); n = r['n']
    c = P[:, 0]; lag = -np.median(P[:, 1::2], axis=1)  # delta = -lag
    base = np.median(lag); lev = np.round(lag - base)  # integer level per window
    # segments: change where integer level changes and stays >= 6 windows (12 frames)
    segs = []; i = 0
    while i < len(lev):
        j = i
        while j < len(lev) and lev[j] == lev[i]: j += 1
        segs.append([i, j, lev[i]]); i = j
    # drop short segments (transition windows) by merging into neighbours
    segs = [s for s in segs if s[1] - s[0] >= 6] or [[0, len(lev), 0.]]
    d = np.zeros(n); vals = []
    for k, (a, b, l) in enumerate(segs):
        vals.append(float(np.median(lag[a:b])))
    # boundaries at midpoints between segment ends and next starts (window centres)
    bounds = [0] + [int(round((c[segs[k][1] - 1] + c[segs[k + 1][0]]) / 2)) for k in range(len(segs) - 1)] + [n]
    for k in range(len(segs)): d[bounds[k]:bounds[k + 1]] = vals[k]
    key = f"{r['split']}_{r['ep']}"; out[key] = d.round(3).tolist(); summ[key] = dict(values=[round(v, 3) for v in vals], bounds=bounds)
    print(key, summ[key])
json.dump(out, open('../out/sync_delta_t.json', 'w')); json.dump(summ, open('../out/sync_delta_t_summary.json', 'w'), indent=1)
