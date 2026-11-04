import sys, numpy as np
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamF_margins/ep19/code')
from common19F import *
import trimesh
w = sys.argv[1]; p = sys.argv[2]
z = np.load(f'{w}/episode_target/episode_000019/reference_trajectory.npz'); on = list(z['object_names'])
t = np.load(p + '/execution/trajectory.npz'); tn = list(t['object_names'])
V = {n: np.asarray(trimesh.load(f'/data1/SRL_V2D/challenge_reconstruction/final_recon/episode_000019/assets/{n}.obj', force='mesh').vertices) for n in NAMES}
for n in NAMES:
    Tr = to_T(z['object_positions'][:, on.index(n)], z['object_quaternions_wxyz'][:, on.index(n)]); Ta = to_T(t['object_positions'][:, tn.index(n)], t['object_quaternions_wxyz'][:, tn.index(n)])
    lr = [((V[n] @ Tr[k, :3, :3].T + Tr[k, :3, 3])[:, 2].min() * 100) for k in range(len(Tr))]; la = [((V[n] @ Ta[k, :3, :3].T + Ta[k, :3, 3])[:, 2].min() * 100) for k in range(len(Ta))]
    print(n, 'lowest z ref/ach (cm) per 0.5 s:', ' '.join('%.1f/%.1f' % (lr[k], la[k]) for k in range(0, len(Tr), 10)))
