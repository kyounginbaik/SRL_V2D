"""T8 ep19: per-window error budget of a rollout trajectory vs target_final_v5 (score_any-faithful, common19F)."""
import sys, json, numpy as np
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamF_margins/ep19/code')
import common19F as CF
traj = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else None
res, e = CF.compose_score_traj(traj); print('scores', res)
T = len(e['mini_sweeper']['cm_t']); W = 10
print('win(s)      sweeper cm/deg  (contrib cm, excess-over-1.3 contrib)   dustpan cm/deg (contrib, excess)')
rows = []
for s in range(0, T, W):
    r = [f'{s/20:4.1f}-{min(s+W,T)/20:4.1f}']
    for n in CF.NAMES:
        c = e[n]['cm_t'][s:s+W]; d = e[n]['deg_t'][s:s+W]
        r.append(f'{c.mean():5.2f}/{d.mean():5.1f} ({c.sum()/T:5.3f}, {np.maximum(c-1.3,0).sum()/T:5.3f})')
    print('  '.join(r))
for n in CF.NAMES:
    c = e[n]['cm_t']; print(n, 'mean', c.mean().round(3), 'need cut', (c.mean()-1.3).round(3), 'frames>3cm', (c > 3).sum(), 'contrib of frames>3cm', (c[c > 3].sum()/T).round(3),
                            'p50', np.median(c).round(2), 'excess mass over 1.3', (np.maximum(c-1.3,0).sum()/T).round(3))
if out: np.savez(out, **{f'{n}_{k}': e[n][k] for n in CF.NAMES for k in ('cm_t', 'deg_t')})
