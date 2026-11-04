"""Fairness check on a few frames: NEW evaluated with its own camera model vs (unfair) our pinhole; OURS as reference."""
import sys, numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0]); import h2hlib as H
ep = int(sys.argv[1]); frames = [int(x) for x in sys.argv[2].split(',')]
e = H.Ep(ep, nsamp=30000)
for o in e.names:
    for lab, tr, mode in [('OURS', 'ours', 'own'), ('NEW own-cam', 'new', 'own'), ('NEW pinhole', 'new', 'pinhole')]:
        e.cam_mode = mode; r = np.array([[e.evidence(tr, o, f).get(k, np.nan) for k in ('dm', 'md', 'fs', 'iou')] for f in frames])
        print(f'ep{ep} {o:14s} {lab:12s} dm {np.nanmedian(r[:,0]):5.1f} md {np.nanmedian(r[:,1]):5.1f} fs {np.nanmedian(r[:,2]):.2f} iou {np.nanmedian(r[:,3]):.2f}   per-frame iou {np.round(r[:,3],2).tolist()}')
