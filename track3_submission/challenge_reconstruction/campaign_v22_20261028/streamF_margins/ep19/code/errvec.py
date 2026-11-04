import sys, numpy as np
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamF_margins/ep19/code')
from common19F import *
w, p = sys.argv[1], sys.argv[2]
z = np.load(f'{w}/episode_target/episode_000019/reference_trajectory.npz'); on = list(z['object_names']); t = np.load(p + '/execution/trajectory.npz'); tn = list(t['object_names'])
reg = registration(packed_meshes()); an = reg['mini_dust_pan']['candidate_anchor']
Tr = to_T(z['object_positions'][:, on.index('mini_dust_pan')], z['object_quaternions_wxyz'][:, on.index('mini_dust_pan')]); Ta = to_T(t['object_positions'][:, tn.index('mini_dust_pan')], t['object_quaternions_wxyz'][:, tn.index('mini_dust_pan')])
pr = np.einsum('tij,j->ti', Tr[:, :3, :3], an) + Tr[:, :3, 3]; pa = np.einsum('tij,j->ti', Ta[:, :3, :3], an) + Ta[:, :3, 3]; e = (pa - pr) * 100
for k in range(0, 235, 10): print('%.1f  err xyz cm %s  |%.2f|' % (k / 20, np.round(e[k], 2), np.linalg.norm(e[k])))
