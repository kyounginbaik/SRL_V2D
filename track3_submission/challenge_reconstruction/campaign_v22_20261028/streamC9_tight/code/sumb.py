import sys, json
for f in sys.argv[1:]:
    print('==', f)
    for l in open(f):
        if l.startswith('INIT') or l.startswith('FINAL'): print(l.strip()[:220])
        if l.startswith('WIN'):
            r = json.loads(l[4:]); b = r['best'][0]
            print(r['window'], 'kc', r['kc'], 'n', r['n'], 'zero', round(min(r['zero']), 1), 'best', round(b['cum'], 1), [round(x, 2) for x in b['sp']], [round(x, 2) for x in b['cup']], 'nb', r['n_better_than_best_zero'], r['elapsed_s'], b['last'][:60])
