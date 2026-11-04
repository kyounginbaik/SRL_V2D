"""B9: per-frame spoon/cup error and achieved-vs-target (physics world) position/height around the spikes. usage: diag_spike.py ACTOR EVAL_NPZ f0 f1"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); import mres
M = '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamV8_ep32/staging/ep32_wb4r2/components/episode_000032/meshes/episode_000032'
env = mres.Env(dict(ep=32, actor=sys.argv[1], meshes=M)); Q = np.load(sys.argv[2])['Q']; f0, f1 = int(sys.argv[3]), int(sys.argv[4])
e, A = mres.L.per_frame(32, env.poses(Q), env.reg, env.tgt, want_A=True); tp = mres.L.target_phys(32, env.reg, A, env.tgt); P = env.poses(Q)
for n in ('wooden_spoon', 'blue_cup'):
    ca = env.reg[n][0]
    for k in range(f0, f1):
        pa = P[n][k, :3, :3] @ ca + P[n][k, :3, 3]; pt = tp[n][k, :3, :3] @ ca + tp[n][k, :3, 3]
        print(n[:5], k, f'{k/20:.2f}', 'err %.2f/%.1f' % (e[n][0][k], e[n][1][k]), 'd(cm)', np.round((pa - pt) * 100, 2), 'z ach %.2f tgt %.2f' % (pa[2] * 100, pt[2] * 100))
