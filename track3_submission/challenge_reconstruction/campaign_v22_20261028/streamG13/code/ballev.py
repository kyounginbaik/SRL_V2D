"""GT-free 2D gray-ball evidence (spoon_roll_v8 chiral model, 2D only) for spoon hypotheses that use our spoon mesh 5fcd334c.
usage: ballev.py EP HYP1.npz HYP2.npz ..."""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamZ9_anchor_spoon/code')
from z9_anchor import ball_model, ball_px
ep = int(sys.argv[1]); root = Path(f'/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/canonical/evaluation/episode_{ep:06d}')
intr = json.load(open(root / 'intrinsics.json')); K = np.array([[intr['fx'], 0, intr['cx']], [0, intr['fy'], intr['cy']], [0, 0, 1.]])
model, dets = ball_model(root, 'wooden_spoon')
H = {h.split('/')[-1][:-4]: np.load(h, allow_pickle=True) for h in sys.argv[2:]}
B = {}
for k, h in H.items():
    assert str(h['meshes'][list(h['names']).index('wooden_spoon')]).endswith('wooden_spoon.obj')
    s = [str(n) for n in h['names']].index('wooden_spoon'); C = h['camera_from_object'][:, s]
    B[k] = np.array([ball_px(model, (dets[t] if t < len(dets) else None) or [], C[t], K) for t in range(len(C))])
ks = list(B); A = np.stack([B[k] for k in ks], 1); ok = np.isfinite(A).all(1)
print('frames with balls', ok.sum(), 'median px', {k: round(float(np.nanmedian(B[k][ok])), 1) for k in ks})
for a, b in [(0, 60), (60, 100), (100, 140), (140, 180), (180, 220), (220, 260), (260, 300), (300, 348)]:
    o = ok.copy(); o[:a] = False; o[b:] = False
    print(f' f{a}-{b} n={o.sum():3d} ' + ' '.join(f'{k}:{np.median(B[k][o]) if o.any() else np.nan:6.1f}' for k in ks) + (f'  wins {ks[0]} {(A[o,0] < A[o,1]-2).sum()} vs {(A[o,1] < A[o,0]-2).sum()}' if len(ks) > 1 else ''))
np.savez(f'../out/ballev_ep{ep}.npz', **B)
