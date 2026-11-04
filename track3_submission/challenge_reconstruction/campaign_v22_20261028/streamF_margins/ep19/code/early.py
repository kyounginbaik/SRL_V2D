import sys, numpy as np
from scipy.spatial.transform import Rotation as R
for p in sys.argv[1:]:
    t = np.load(p + '/execution/trajectory.npz'); on = list(t['object_names']); i = on.index('mini_sweeper')
    P = t['object_positions'][:, i]; Q = t['object_quaternions_wxyz'][:, i]; r0 = R.from_quat(Q[0][[1, 2, 3, 0]])
    print(p.split('/')[-1], ' '.join('%d:%.1fcm/%.0fd' % (k, np.linalg.norm(P[k] - P[0]) * 100, np.degrees((R.from_quat(Q[k][[1, 2, 3, 0]]) * r0.inv()).magnitude())) for k in [1, 5, 10, 15, 20, 25, 30, 35, 40]))
