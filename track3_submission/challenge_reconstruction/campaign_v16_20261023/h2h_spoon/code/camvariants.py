"""Fairness check: which camera mapping makes the NEW track fit the evidence best? variants of camera_a_from_obj."""
import sys, numpy as np
sys.path.insert(0, '.'); import h2hlib as H
ep = int(sys.argv[1]); frames = [int(x) for x in sys.argv[2].split(',')]
e = H.Ep(ep, nsamp=30000); Ta = e.Ta; Ti = np.linalg.inv(Ta)
V = {'Ta.inv(c).T': lambda c, T: Ta @ np.linalg.inv(c) @ T, 'inv(c).T': lambda c, T: np.linalg.inv(c) @ T,
     'Ta^-1.inv(c).T': lambda c, T: Ti @ np.linalg.inv(c) @ T, 'inv(Ta c Ta^-1).Ta.T': lambda c, T: np.linalg.inv(Ta @ c @ Ti) @ Ta @ T}
for o in e.names:
    for name, fn in [('OURS', None)] + list(V.items()):
        res = []
        for f in frames:
            if fn is None: r = e.evidence('ours', o, f)
            else:
                orig = e.cam_T; e.cam_T = lambda tr, oo, ff, fn=fn: fn(e.c0Tc[ff], e.Tnew[oo][ff]); r = e.evidence('new', o, f); e.cam_T = orig
            res.append((r.get('dm', np.nan), r.get('fs', np.nan), r['iou']))
        res = np.array(res); print(f'ep{ep} {o:14s} {name:22s} dm {np.nanmedian(res[:,0]):6.1f}  fs {np.nanmedian(res[:,1]):.2f}  iou {np.nanmedian(res[:,2]):.2f}')
