"""T33: one line per CEM iteration (fcem33 / lcem33 logs). usage: prog33.py LOG ..."""
import sys, json
for f in sys.argv[1:]:
    print('==', f)
    for l in open(f):
        if l.startswith('RESULT') or l.startswith('MADE'): print(l.strip()[:400]); continue
        if not l.startswith('IT '): continue
        h = json.loads(l[3:]); b = h['best']
        print(h['iteration'], round(b['cost'], 3), {k: (round(v[0], 2), round(v[1], 1)) for k, v in b['objects'].items()}, {k: round(v, 2) for k, v in (b.get('grasp') or b.get('left') or {}).items()}, h.get('n_pass'), int(h['elapsed_s']))
