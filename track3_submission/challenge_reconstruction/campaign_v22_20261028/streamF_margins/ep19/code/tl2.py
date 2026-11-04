import sys, numpy as np
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamF_margins/ep19/code')
from common19F import *
for p in sys.argv[1:]:
    cs, e = compose_score_traj(p); print(p.split('/')[-4] if 'actor' in p else p.split('/')[-3], cs)
    for n in NAMES: print('  ', n[5:9], ' '.join('%.1f' % e[n]['cm_t'][k:k + 10].mean() for k in range(0, 235, 10)))
