"""Hand-object grip check: change of the object pose in the carrying wrist frame vs a reference time (cm / deg)."""
import sys, numpy as np
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamV_C19/code')
from common19 import *
WC = {'left': [0, 1, 2, 3, 4, 5], 'right': [28, 29, 30, 31, 32, 33]}
def grip(traj, obj, side, t_ref=2.0, times=None):
    z = np.load(traj); i = list(z['object_names']).index(obj); q = z['qpos']; n = len(z['object_positions'])
    To = to_T(z['object_positions'][:, i], z['object_quaternions_wxyz'][:, i])
    Tw = np.tile(np.eye(4), (n, 1, 1)); Tw[:, :3, 3] = q[:n, WC[side][:3]]; Tw[:, :3, :3] = R.from_euler('XYZ', q[:n, WC[side][3:]]).as_matrix()
    rel = np.einsum('tji,tjk->tik', Tw, To) if False else np.linalg.inv(Tw) @ To; k0 = int(t_ref * 20); out = []
    for t in (times or np.arange(0, n / 20, 0.5)):
        k = min(int(t * 20), n - 1); D = np.linalg.inv(rel[k0]) @ rel[k]; out.append((t, round(float(np.linalg.norm(D[:3, 3]) * 100), 1), round(float(np.degrees(R.from_matrix(D[:3, :3]).magnitude())))))
    return out
if __name__ == '__main__':
    for d in sys.argv[3:]: print(d.split('/')[-1], sys.argv[1], grip(d + '/execution/trajectory.npz', sys.argv[1], sys.argv[2]))
