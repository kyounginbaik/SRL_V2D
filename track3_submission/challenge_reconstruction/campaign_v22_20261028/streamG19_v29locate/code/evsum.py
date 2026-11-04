"""Summarise ev.py outputs: per ep/object/hypothesis mean IoU, median signed / abs depth residual (mm), and on the frames where the
hypothesis differs from v14 by > 3 cm or > 30 deg (camera frame, gross-disagreement window): IoU and |dz| of v14 vs hypothesis."""
import json, os, sys, numpy as np
from scipy.spatial.transform import Rotation as Rot
S = '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG19_v29locate'
def nm(x): return np.array([np.nan if v is None else v for v in x], float)
out = {}
for ep in (3, 33, 35, 36, 20, 1, 19):
    base = json.load(open(f'{S}/out/ev/ep{ep}_v14.json'))['per_frame']; Hb = np.load(f'{S}/out/hyp/ep{ep}_v14.npz'); names = [str(n) for n in Hb['names']]
    for h in ('v29', 'fro', 'c21spoon'):
        fn = f'{S}/out/ev/ep{ep}_{h}.json'
        if not os.path.exists(fn): continue
        d = json.load(open(fn))['per_frame']; Hh = np.load(f'{S}/out/hyp/ep{ep}_{h}.npz')
        A, B = Hb['camera_from_object'], Hh['camera_from_object']; F = min(len(A), len(B))
        for s, n in enumerate(names):
            if h == 'c21spoon' and n != 'wooden_spoon': continue
            dp = np.linalg.norm(A[:F, s, :3, 3] - B[:F, s, :3, 3], axis=-1) * 100
            dr = np.degrees(np.linalg.norm(Rot.from_matrix(np.einsum('fji,fjk->fik', A[:F, s, :3, :3], B[:F, s, :3, :3])).as_rotvec(), axis=-1))
            ib, ih = nm(base[n]['iou'])[:F], nm(d[n]['iou'])[:F]; zb, zh = nm(base[n]['adz'])[:F], nm(d[n]['adz'])[:F]; sb, sh = nm(base[n]['dz'])[:F], nm(d[n]['dz'])[:F]
            win = (dp > 3) | (dr > 30); ok = win & np.isfinite(ib) & np.isfinite(ih)
            r = dict(frames=int(F), mean_dpos_cm=round(float(dp.mean()), 2), mean_drot_deg=round(float(dr.mean()), 1), gross_frames=int(win.sum()),
                     iou_v14=round(float(np.nanmean(ib)), 3), iou_hyp=round(float(np.nanmean(ih)), 3),
                     dz_signed_v14=round(float(np.nanmedian(sb)), 1), dz_signed_hyp=round(float(np.nanmedian(sh)), 1),
                     adz_v14=round(float(np.nanmedian(zb)), 1), adz_hyp=round(float(np.nanmedian(zh)), 1),
                     gross_iou_v14=round(float(np.nanmean(ib[ok])), 3) if ok.any() else None, gross_iou_hyp=round(float(np.nanmean(ih[ok])), 3) if ok.any() else None,
                     gross_frames_hyp_better_iou=int((ih[ok] > ib[ok] + 0.02).sum()), gross_frames_v14_better_iou=int((ib[ok] > ih[ok] + 0.02).sum()))
            out[f'{ep}/{n}/{h}'] = r; print(ep, n, h, r)
json.dump(out, open(f'{S}/out/ev_summary.json', 'w'), indent=1)
