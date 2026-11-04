"""Stream R ep32: per-frame roll of the reference spoon relative to the mocap balls (input: spoon_markers.py JSON).
Balls are clustered by their position along the spoon long axis (body x) into the three physical markers using the rest frames
0-44 (spoon static on the table) as the canonical body coordinates; per detection the roll deviation = signed angle about body x
between the observed perpendicular offset and the canonical one; per frame the median over the matched balls.
Also a translation-free variant from ball PAIRS (angle of the pair vector about x).
usage: marker_roll.py MARKERS_JSON [OUT_JSON]"""
import sys, json, numpy as np
d = json.load(open(sys.argv[1])); rows = d['rows']
pts = [(r['frame'], np.array(x['body_m'])) for r in rows for x in r['detections']]
ok = lambda b: abs(b[0]) < 0.17 and np.hypot(b[1], b[2]) < 0.045 and np.hypot(b[1], b[2]) > 0.008
rest = np.array([b for f, b in pts if f <= 44 and ok(b)])
# cluster along x (k-means 1D, 3 clusters)
c = np.percentile(rest[:, 0], [10, 50, 90])
for _ in range(20):
    lab = np.argmin(abs(rest[:, [0]] - c[None]), 1); c = np.array([np.median(rest[lab == j, 0]) if (lab == j).any() else c[j] for j in range(3)])
canon = [np.median(rest[lab == j], 0) for j in range(3)]
print('canonical ball body coords (cm):', [np.round(x * 100, 1).tolist() for x in canon], 'counts', [(lab == j).sum() for j in range(3)])
ang = lambda b: np.degrees(np.arctan2(b[2], b[1]))
wrap = lambda a: (a + 180) % 360 - 180
out = []
for r in rows:
    m = []
    for x in r['detections']:
        b = np.array(x['body_m'])
        if not ok(b): continue
        j = int(np.argmin([abs(b[0] - cc[0]) for cc in canon]))
        if abs(b[0] - canon[j][0]) > 0.025: continue
        m.append((j, b, wrap(ang(b) - ang(canon[j]))))
    roll = float(np.median([x[2] for x in m])) if m else None
    pair = []
    for i in range(len(m)):
        for k in range(i + 1, len(m)):
            if m[i][0] == m[k][0]: continue
            vo = m[k][1] - m[i][1]; vc = canon[m[k][0]] - canon[m[i][0]]
            if np.hypot(vc[1], vc[2]) > 0.015: pair.append(wrap(ang(vo) - ang(vc)))
    out.append(dict(frame=r['frame'], n=len(m), roll_deg=roll, pair_roll_deg=float(np.median(pair)) if pair else None, perp_cm=[float(np.hypot(x[1][1], x[1][2]) * 100) for x in m]))
print('  t   | n  roll(single) pair-roll  (reference minus balls: + = reference rolled positive about body x)')
for s in range(0, len(out), 10):
    g = out[s:s+10]; sr = [x['roll_deg'] for x in g if x['roll_deg'] is not None]; pr = [x['pair_roll_deg'] for x in g if x['pair_roll_deg'] is not None]
    print(f"{s/20:5.1f} | {np.mean([x['n'] for x in g]):3.1f} {np.median(sr) if sr else float('nan'):7.1f} ({len(sr):2d}) {np.median(pr) if pr else float('nan'):7.1f} ({len(pr):2d})")
if len(sys.argv) > 2: json.dump(dict(canonical_body_m=[x.tolist() for x in canon], rows=out), open(sys.argv[2], 'w'), indent=1)
