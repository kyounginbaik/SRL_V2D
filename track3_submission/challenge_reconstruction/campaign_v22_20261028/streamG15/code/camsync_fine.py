"""GT-free per-frame sync profile: lag of ego_cam_b/c vs ego_cam_a in sliding windows (24 frames, step 2).
Usage camsync_fine.py SPLIT EP -> out/syncfine_<split>_<ep>.json"""
import sys, json, numpy as np
sys.argv = sys.argv[:3]
import importlib.util
src = open('camsync.py').read().split("S = {c:")[0]
exec(src)
S = {c: motion(c) for c in ['ego_cam_a', 'ego_cam_b', 'ego_cam_c']}
def xc(a, b, maxlag=3, sub=0.05):
    t = np.arange(len(a)); best = (0, -2)
    for L in np.arange(-maxlag, maxlag + 1e-9, sub):
        bb = np.interp(t + L, t, b); m = slice(maxlag + 1, len(a) - maxlag - 1)
        aa = a[m] - a[m].mean(); cc = bb[m] - bb[m].mean(); r = (aa * cc).sum() / np.sqrt((aa ** 2).sum() * (cc ** 2).sum() + 1e-12)
        if r > best[1]: best = (L, r)
    return best
n = min(len(v) for v in S.values()); prof = []
for st in range(0, n - 24, 2):
    sl = slice(st, st + 24); row = [st + 12]
    for c in ['ego_cam_b', 'ego_cam_c']:
        for j in range(2):
            L, r = xc(S['ego_cam_a'][sl, j], S[c][sl, j]); row += [L, r]
    prof.append(row)
prof = np.array(prof); json.dump(dict(ep=ep, split=split, n=n, prof=prof.tolist()), open(f'../out/syncfine_{split}_{ep}.json', 'w'))
for r in prof[::3]: print(int(r[0]), ' '.join('%.2f' % x for x in r[1::2]), ' r', ' '.join('%.2f' % x for x in r[2::2]))
