"""Absolute-depth check independent of either stereo pipeline: the spoon mesh is metric (30.8 cm), so its projected size fixes
its distance. For rest frames, slide each track's pose along the camera ray through the object (C.t *= s, orientation kept) and
find the s that maximises silhouette IoU with the SAM2 mask (hand ignored). s* ~ 1 means the track's depth is consistent
with the object's true size. Also the stereo depth ratio median(stereo z / model z) over the visible surface for reference.
usage: scale_scan.py EP OBJ F0 F1 STEP"""
import sys, numpy as np, cv2
sys.path.insert(0, __file__.rsplit('/', 1)[0]); import h2hlib as H
ep, obj, f0, f1, st = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
e = H.Ep(ep, nsamp=120000); S = np.linspace(0.90, 1.10, 41)
for tr in ('ours', 'new'):
    best = []; ratio = []
    for f in range(f0, f1 + 1, st):
        objm = e.mask(obj, f); hand = cv2.dilate(e.hand(f).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0; C0 = e.cam_T(tr, obj, f); ious = []
        for s in S:
            C = C0.copy(); C[:3, 3] *= s; z = e.zbuf((tr, obj), C, e.model_of(tr)); r = np.isfinite(z)
            r = cv2.morphologyEx(r.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
            ious.append(((r & objm & ~hand).sum()) / max(((r | objm) & ~hand).sum(), 1))
        ious = np.array(ious); best.append(S[ious.argmax()])
        z = e.zbuf((tr, obj), C0, e.model_of(tr)); dep = e.depth(f); v = np.isfinite(z) & np.isfinite(dep) & objm & ~hand
        ratio.append(np.median(dep[v] / z[v]) if v.sum() > 50 else np.nan)
    print(f'ep{ep} {obj} {tr:4s} frames {f0}-{f1}: IoU-optimal depth scale s* median {np.median(best):.3f} (IQR {np.percentile(best,25):.3f}-{np.percentile(best,75):.3f}); stereo/model depth ratio {np.nanmedian(ratio):.3f}', flush=True)
