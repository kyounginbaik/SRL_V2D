"""Compare two ev.py outputs (same episode) on moving frames, split at a frame: signed/abs stereo depth residual and IoU.
Stereo depth is on the b/c clock, masks on the RGB clock (G15). usage: evcmp.py EV_A EV_B BUNDLE_A EP SPLIT_FRAME"""
import sys, json, numpy as np
A, B, BA, ep, k = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
W = np.load(f'{BA}/episode_{ep:06d}/object_poses.npz')['world_from_object']
ea, eb = json.load(open(A))['per_frame'], json.load(open(B))['per_frame']
arr = lambda r, key: np.array([np.nan if x is None else x for x in r[key]], float)
for s, n in enumerate(ea):
    sp = np.r_[0, np.linalg.norm(np.diff(W[:, s, :3, 3], axis=0), axis=1) * 100]
    for lab, sel in (('f0-%d' % (k - 1), np.arange(len(sp)) < k), ('f%d-end' % k, np.arange(len(sp)) >= k)):
        for mlab, mv in (('moving>0.5cm/f', sp > 0.5), ('all', sp >= 0)):
            m = sel & mv
            ad_a, ad_b = arr(ea[n], 'adz')[m], arr(eb[n], 'adz')[m]
            ok = np.isfinite(ad_a) & np.isfinite(ad_b)
            if ok.sum() < 3: continue
            ia, ib = arr(ea[n], 'iou')[m][ok], arr(eb[n], 'iou')[m][ok]
            win = np.mean(ad_b[ok] < ad_a[ok])
            print(f'{n:14s} {lab:9s} {mlab:15s} n{ok.sum():3d} |dz| mm A {np.median(ad_a[ok]):6.2f} B {np.median(ad_b[ok]):6.2f}  B better on {win:.0%} | '
                  f'IoU A {np.median(ia):.3f} B {np.median(ib):.3f}')
