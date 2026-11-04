"""Per-frame gap: v21 rollout and our reference vs compose (in our body frames, G = pot frame-0 anchor alignment onto our reference).
usage: analyze_gap.py EP RESULTS_PARQUET"""
import sys, numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0]); import tlib as L
ep, res = int(sys.argv[1]), sys.argv[2]
Tc, X = L.compose_ourbody(ep); A, B = L.results_tracks(res); anc = L.anchor_points(ep)
n = min(len(A['white_pot']), len(Tc['white_pot']))
G = B['white_pot'][0] @ np.linalg.inv(Tc['white_pot'][0])
for o in L.NAMES:
    Cw = G @ Tc[o][:n]
    pa, ra = L.errs(A[o][:n], Cw, anc[o]); pb, rb = L.errs(B[o][:n], Cw, anc[o]); pab, rab = L.errs(A[o][:n], B[o][:n], anc[o])
    print(f'ep{ep} {o:13s} rollout-vs-compose {pa.mean():.2f}/{ra.mean():.1f}  ourref-vs-compose {pb.mean():.2f}/{rb.mean():.1f}  rollout-vs-ourref {pab.mean():.2f}/{rab.mean():.1f}')
    for s in range(0, n, 20):
        e = min(s + 20, n); print(f'   f{s:3d}-{e-1:3d} roll/cmp {pa[s:e].mean():5.2f} {ra[s:e].mean():5.1f} | ref/cmp {pb[s:e].mean():5.2f} {rb[s:e].mean():5.1f} | z ref {B[o][s:e,2,3].mean()*100:6.2f} cmp {Cw[s:e,2,3].mean()*100:6.2f}')
