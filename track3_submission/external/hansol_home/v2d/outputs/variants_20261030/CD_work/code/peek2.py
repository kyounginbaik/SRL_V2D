import json, sys, numpy as np
np.set_printoptions(linewidth=220, suppress=True, precision=4)
obj = sys.argv[1]
for fn in sys.argv[2:]:
    d = json.load(open(fn))
    if obj not in d: continue
    s = d[obj]['summary']; R = np.array(d[obj]['rows'], float)
    print('==', fn.split('/')[-1], {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()})
    # columns: frame, s_hat, r, a?, z, dz_m, iou, npx
    print(' frames', R[:, 0].astype(int))
    print(' r     ', R[:, 2])
    print(' dz_mm ', np.round(R[:, 5] * 1000, 1))
    print(' z     ', R[:, 4])
