"""Insertion depth (blue rim z - beige bottom z, cm) and tilts over time for one or more trajectory.npz files, using the new
(scan) meshes; plus the tracking reference with the same meshes. usage: insertion_of.py TRAJ [TRAJ...]"""
import sys, numpy as np, trimesh
from scipy.spatial.transform import Rotation as R
EP10 = '/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep10'; S = EP10 + '/source/episode_000010'
Mv = {n: np.asarray(trimesh.load(f'{S}/assets/{n}_visual.obj', force='mesh').vertices) for n in ('blue_cup', 'beige_cup')}
z = np.load(S + '/object_poses.npz'); names = [str(x) for x in z['names']]; T = z['world_from_object']
ts = (3, 4, 5, 5.5, 6, 7, 8)
def ins(Rb, pb, Rt, pt):
    bl = Mv['blue_cup'] @ Rb.T + pb; be = Mv['beige_cup'] @ Rt.T + pt; return (bl[:, 2].max() - be[:, 2].min()) * 100, be[:, 2].max() * 100
print('reference :', ' '.join(f'{t}:{ins(T[int(t*20),names.index("blue_cup"),:3,:3],T[int(t*20),names.index("blue_cup"),:3,3],T[int(t*20),names.index("beige_cup"),:3,:3],T[int(t*20),names.index("beige_cup"),:3,3])[0]:.1f}' for t in ts))
for p in sys.argv[1:]:
    a = np.load(p); on = list(a['object_names']); q = a['object_quaternions_wxyz'][..., [1, 2, 3, 0]]; P = a['object_positions']; n = len(P)
    vals = []
    for t in ts:
        k = min(int(t * 20), n - 1); ib, it = on.index('blue_cup'), on.index('beige_cup')
        vals.append(f'{t}:{ins(R.from_quat(q[k, ib]).as_matrix(), P[k, ib], R.from_quat(q[k, it]).as_matrix(), P[k, it])[0]:.1f}')
    print(p.split('refinement_20261021_ep10/')[-1][:60].ljust(60), ' '.join(vals))
