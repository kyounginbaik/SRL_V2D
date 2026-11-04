import sys, numpy as np
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamF_margins/ep19/code')
import common19F as CF
traj = sys.argv[1]; n = sys.argv[2]; f0, f1 = int(sys.argv[3]), int(sys.argv[4])
if CF._REG is None: CF._REG = CF.registration(CF.packed_meshes())
t = np.load(traj); on = list(t['object_names']); comp = CF.compose_tracks(); reg = CF._REG
ach = {m: CF.to_T(t['object_positions'][:, on.index(m)], t['object_quaternions_wxyz'][:, on.index(m)]) for m in CF.NAMES}
# replicate alignment to get vectors
A, Rf = {}, {}
for m in CF.NAMES:
    r = reg[m]; a = ach[m].copy(); a[:, :3, 3] += np.einsum('tij,j->ti', a[:, :3, :3], r['candidate_anchor']); a[:, :3, :3] = a[:, :3, :3] @ r['rot'].T
    b = comp[m].copy(); b[:, :3, 3] += np.einsum('tij,j->ti', b[:, :3, :3], r['anchor']); A[m], Rf[m] = a, b
Ra = Rf[CF.NAMES[0]][0, :3, :3] @ A[CF.NAMES[0]][0, :3, :3].T; off = Rf[CF.NAMES[0]][0, :3, 3] - Ra @ A[CF.NAMES[0]][0, :3, 3]
p = A[n][:, :3, 3] @ Ra.T + off; q = Rf[n][:len(p), :3, 3]
from scipy.spatial.transform import Rotation as R
Rr = np.einsum('ij,tjk->tik', Ra, A[n][:, :3, :3])
for f in range(f0, min(f1, len(p))):
    d = (p[f] - q[f]) * 100; ang = np.degrees(R.from_matrix(Rr[f].T @ Rf[n][f, :3, :3]).magnitude())
    print(f'f{f:3d} {f/20:5.2f}s err {np.linalg.norm(d):5.2f} d=({d[0]:6.2f},{d[1]:6.2f},{d[2]:6.2f}) deg {ang:5.1f}  ach z {p[f,2]*100:6.2f} tgt z {q[f,2]*100:6.2f}  tgt v {np.linalg.norm(q[f]-q[f-1])*2000 if f else 0:5.1f}cm/s ach v {np.linalg.norm(p[f]-p[f-1])*2000 if f else 0:5.1f}')
