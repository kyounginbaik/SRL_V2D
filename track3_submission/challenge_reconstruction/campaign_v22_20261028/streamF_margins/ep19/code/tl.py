"""timeline of scorer errors + object heights for port dirs"""
import sys, numpy as np
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamF_margins/ep19/code')
from common19F import *
step = 10
for p in sys.argv[1:]:
    cs, e = compose_score_traj(p + '/execution/trajectory.npz'); t = np.load(p + '/execution/trajectory.npz'); on = list(t['object_names'])
    print(p.split('/')[-1], cs)
    for k in range(0, 235, step): print('  %.1f' % (k / 20), ' '.join('%s %.1f/%2.0f z%.1f' % (n[5:9], e[n]['cm_t'][k], e[n]['deg_t'][k], t['object_positions'][k, on.index(n), 2] * 100) for n in NAMES))
