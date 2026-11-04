"""J8 ep19: rank distilled actors (attempts.json) vs v22b F_ep19_twin3 (sweeper 1.95/14.8, dust pan 2.38/9.9)."""
import json, sys, glob
V = {'mini_sweeper': (1.95, 14.8), 'mini_dust_pan': (2.38, 9.9)}
rows = []
for f in sys.argv[1:]:
    for g in glob.glob(f + '/attempts.json'):
        for r in json.load(open(g)):
            e = r['object_errors']; better = all(e[k]['cm'] < V[k][0] and e[k]['deg'] < V[k][1] for k in V)
            score = max(max(e[k]['cm'] / 1.2, e[k]['deg'] / 12) for k in V) + 0.25 * sum(max(e[k]['cm'] / 1.2, e[k]['deg'] / 12) for k in V)
            rows.append((score, better, r['checkpoint'].split('/')[-4] + '/' + r['checkpoint'].split('/')[-3], {k[5:]: (e[k]['cm'], e[k]['deg']) for k in V}))
for r in sorted(rows)[:int(20)]: print('%.3f %s %-45s %s' % r)
