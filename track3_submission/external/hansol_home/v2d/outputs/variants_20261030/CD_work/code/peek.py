import json, sys, numpy as np
np.set_printoptions(linewidth=200, suppress=True)
SEG = int(sys.argv[1])
for fn in sys.argv[2:]:
    d = json.load(open(fn))
    print('==', fn.split('/')[-1])
    for n, r in d['per_frame'].items():
        for k in ('iou', 'dz', 'adz', 'dcx', 'dcy'):
            if k not in r: continue
            a = np.array([np.nan if x is None else x for x in r[k]], float)
            segs = [np.nanmedian(a[i:i + SEG]) if np.isfinite(a[i:i + SEG]).any() else np.nan for i in range(0, len(a), SEG)]
            print(f'{n:14s} {k:4s} n{len(a)} med {np.nanmedian(a):7.3f} f0-4 {np.round(a[:5], 3)} seg{SEG} {np.round(segs, 3)}')
