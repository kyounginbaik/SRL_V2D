"""G16 analysis of g16scan outputs (GT-free). Per episode:
 - constant ball-model offset OFF (best roll on static informative frames f0-23, flip 0) -> frame-0 roll consistency check across eps
 - per-frame single-frame best hypothesis (flip, roll) vs adopted (flip 0, roll OFF): disagreement > 60 deg on informative frames
 - Viterbi over all (flip, roll) hypotheses (unary = min(px,40)/4 on informative frames, transition LAM * geodesic deg)
 - windows where the Viterbi path is > 60 deg from adopted; ball px before (adopted @OFF, and raw @0) and after (path)
 - end-flip (tip<->handle) evidence: balls (best flip1 vs best flip0) and silhouette IoU / depth (flip1 best vs flip0 @ nearest OFF)
usage: g16ana.py EP [LAM=0.05] [PXMAX=20] [MINB=2] [ROLLONLY=1]
ROLLONLY=1: the Viterbi / single-frame best use flip-0 (roll-only) hypotheses; informative = best flip-0 px < PXMAX.
(End-flip hypotheses were found to fit spurious in-hand detections 100-160 px away from the adopted balls while the
silhouette rejects them by 0.2-0.3 IoU; the end flip is judged separately from balls AND IoU.)"""
import sys, glob, json, numpy as np
from scipy.spatial.transform import Rotation as Rot
G = '/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/streamG16_spoon'
EP = int(sys.argv[1]); LAM = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
PXMAX = float(sys.argv[3]) if len(sys.argv) > 3 else 20.; MINB = int(sys.argv[4]) if len(sys.argv) > 4 else 2
ROLLONLY = int(sys.argv[5]) if len(sys.argv) > 5 else 1
fs = sorted(glob.glob(f'{G}/out/scan_ep{EP}_*.npz')); Z = [np.load(f) for f in fs]
cat = lambda k: np.concatenate([z[k] for z in Z])
frames, ball, iou, dz, nb, hand, mpx = (cat(k) for k in ('frames', 'ball', 'iou', 'dz', 'nb', 'hand', 'mask_px'))
assert (np.diff(frames) == 1).all() and frames[0] == 0
thb, thi, ax1, ax2 = Z[0]['thb'], Z[0]['thi'], Z[0]['axis'], Z[0]['axis2']
nf, nt = len(frames), len(thb)
wrap = lambda a: (np.asarray(a) + 180) % 360 - 180


def R(flip, th):
    r = Rot.from_rotvec(np.radians(th) * ax1)
    return r * Rot.from_rotvec(np.pi * ax2) if flip else r


H = [(f, t) for f in (0, 1) for t in thb]; RH = Rot.concatenate([R(f, t) for f, t in H])
Q = RH.as_quat(); D = np.degrees(2 * np.arccos(np.clip(np.abs(Q @ Q.T), 0, 1)))   # pairwise geodesic deg
B = ball.reshape(nf, -1).copy()                                                    # (nf, 2*nt) in H order
if ROLLONLY: B[:, nt:] = np.nan
bestpx = np.min(np.where(np.isnan(B), np.inf, B), 1)
inf = (nb >= MINB) & (bestpx < PXMAX)
# constant ball-model offset from early static informative frames (flip 0)
e0 = [f for f in range(min(24, nf)) if inf[f]]
OFF = float(np.median(wrap(thb[np.nanargmin(ball[e0, 0], 1)]))) if e0 else 0.0
jA = int(np.argmin(np.abs(wrap(thb - OFF))))                 # adopted hypothesis (flip 0, roll OFF)
dA = D[jA]                                                   # angle of every hypothesis from adopted
# single-frame best
sb = np.array([int(np.nanargmin(B[f])) if inf[f] else jA for f in range(nf)])
dis1 = inf & (dA[sb] > 60)
marg = np.array([B[f, jA] - B[f, sb[f]] if inf[f] else 0. for f in range(nf)])
# Viterbi
U = np.zeros((nf, len(H)))
for f in range(nf):
    if inf[f]: U[f] = np.minimum(np.nan_to_num(B[f], nan=40.), 40.) / 4.
cost = U[0].copy(); bp = np.zeros((nf, len(H)), int); T = LAM * D
if ROLLONLY: T = T.copy(); T[:, nt:] = 1e9
for f in range(1, nf):
    tot = cost[:, None] + T; bp[f] = np.argmin(tot, 0); cost = tot[bp[f], np.arange(len(H))] + U[f]
path = np.zeros(nf, int); path[-1] = int(np.argmin(cost))
for f in range(nf - 1, 0, -1): path[f - 1] = bp[f, path[f]]
pdev = dA[path]; pflip = np.array([H[j][0] for j in path]); proll = wrap(np.array([H[j][1] for j in path]) - OFF)
segs = []; t = 0
while t < nf:
    if pdev[t] > 60:
        a = t
        while t < nf and pdev[t] > 60: t += 1
        segs.append(dict(f0=a, f1=t - 1, n=t - a, flip=int(np.round(pflip[a:t].mean())), roll_med=float(np.median(proll[a:t])),
                         informative=int(inf[a:t].sum()), sf_disagree=int(dis1[a:t].sum()), px_adopted=float(np.median(B[a:t, jA][inf[a:t]])) if inf[a:t].any() else None,
                         px_path=float(np.median(B[np.arange(a, t), path[a:t]][inf[a:t]])) if inf[a:t].any() else None))
    else: t += 1
pxA = B[inf, jA]; px0 = B[inf, int(np.argmin(np.abs(wrap(thb))))]; pxP = B[np.arange(nf), path][inf]
# end-flip evidence
bf0 = np.nanmin(ball[:, 0], 1); bf1 = np.nanmin(ball[:, 1], 1)
ji = int(np.argmin(np.abs(wrap(thi - OFF))))
hasm = mpx > 30
iA = iou[:, 0, ji]; i1 = np.nanmax(iou[:, 1], 1); i0b = np.nanmax(iou[:, 0], 1)
dzA = dz[:, 0, ji]; dz1 = np.array([dz[f, 1, np.nanargmax(iou[f, 1])] if np.isfinite(iou[f, 1]).any() else np.nan for f in range(nf)])
flipball = inf & (bf1 < bf0 - 3)
flipiou = hasm & (i1 > iA + 0.05)
# frame-0 window (static frames)
f0w = dict(best_roll_f0=float(wrap(thb[np.nanargmin(ball[0, 0])] - OFF)) if inf[0] else None, px_f0_adopted=float(B[0, jA]) if inf[0] else None,
           iou_f0_adopted=float(iA[0]) if hasm[0] else None, iou_f0_best_flip0=float(i0b[0]) if hasm[0] else None,
           iou_f0_best_flip1=float(i1[0]) if hasm[0] else None, informative_in_f0_23=len(e0),
           best_rolls_f0_23=[float(wrap(thb[np.nanargmin(ball[f, 0])])) for f in e0])
out = dict(ep=EP, lam=LAM, pxmax=PXMAX, minb=MINB, nf=nf, informative=int(inf.sum()), offset_deg=OFF,
           sf_disagree_frac_of_informative=round(float(dis1.sum() / max(1, inf.sum())), 3), sf_disagree_frames=int(dis1.sum()),
           sf_disagree_strong=int((dis1 & (marg > 5)).sum()),
           path_dev60_frac_all=round(float((pdev > 60).mean()), 3), path_dev30_frac_all=round(float((pdev > 30).mean()), 3),
           path_flip_frames=int(pflip.sum()), windows=segs,
           px_median=dict(adopted_at_off=float(np.median(pxA)) if len(pxA) else None, adopted_raw=float(np.median(px0)) if len(px0) else None,
                          path=float(np.median(pxP)) if len(pxP) else None),
           px_mean=dict(adopted_at_off=float(np.mean(np.minimum(pxA, 40))) if len(pxA) else None, path=float(np.mean(np.minimum(pxP, 40))) if len(pxP) else None),
           endflip=dict(ball_frames_flip_better_by3=int(flipball.sum()), ball_best_flip0_med=float(np.nanmedian(bf0[inf])) if inf.any() else None,
                        ball_best_flip1_med=float(np.nanmedian(bf1[inf])) if inf.any() else None,
                        mask_frames=int(hasm.sum()), iou_flip1_better_by005=int(flipiou.sum()), iou_adopted_med=float(np.nanmedian(iA[hasm])),
                        iou_best_flip1_med=float(np.nanmedian(i1[hasm])), iou_best_flip0_med=float(np.nanmedian(i0b[hasm])),
                        dz_adopted_med=float(np.nanmedian(dzA[hasm])), dz_flip1_med=float(np.nanmedian(dz1[hasm]))),
           frame0=f0w)
json.dump(out, open(f'{G}/out/ana_ep{EP}_l{LAM}_p{PXMAX:g}_b{MINB}_r{ROLLONLY}.json', 'w'), indent=1)
np.savez(f'{G}/out/ana_ep{EP}_l{LAM}_p{PXMAX:g}_b{MINB}_r{ROLLONLY}.npz', path=path, pdev=pdev, pflip=pflip, proll=proll, inf=inf, sb=sb, marg=marg, offset=OFF,
         jA=jA, H=np.array(H), iouA=iA, iou1=i1)
print(EP, 'OFF', OFF, 'inf', int(inf.sum()), '/', nf, 'sf>60', out['sf_disagree_frames'], '(strong', out['sf_disagree_strong'], ') path>60', out['path_dev60_frac_all'],
      'px A/raw/path', out['px_median'], 'flip:', out['endflip']['ball_frames_flip_better_by3'], out['endflip']['iou_flip1_better_by005'], '/', int(hasm.sum()))
for s in segs: print('   win', s)
