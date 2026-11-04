import json, sys
for f in sys.argv[1:]:
    print('==', f)
    for l in open(f):
        if l.startswith('WIN'):
            r = json.loads(l[4:]); b = r['best'][0]
            print(r['window'], 'n', r['n'], 'zero', [round(z, 1) for z in r['zero']][:3], 'best', round(b['cum'], 1), 'sp', [round(x, 2) for x in b['sp']], 'cup', [round(x, 2) for x in b['cup']], 'nb', r['n_better_than_best_zero'], r['elapsed_s'], b['last'][:60])
        elif l.startswith(('INIT', 'FINAL')) or 'Error' in l or 'rror' in l[:40]: print(l.strip()[:300])
