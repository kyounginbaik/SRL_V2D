"""V8: error structure of an ep32 rollout (eval.npz Q): world pos error vector, rotation error rotvec in the spoon body frame,
spoon relative to the cup (target vs achieved), right wrist vs spoon. usage: diag.py ACTOR MESHES EVALNPZ"""
import sys, numpy as np
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamM8_32_33_15_16/code')
import mres
from scipy.spatial.transform import Rotation as R
env = mres.Env(dict(ep=32, actor=sys.argv[1], meshes=sys.argv[2])); Q = np.load(sys.argv[3])['Q']
P = env.poses(Q); e, A = mres.L.per_frame(32, P, env.reg, env.tgt, want_A=True); tp = mres.L.target_phys(32, env.reg, A, env.tgt)
s, c = 'wooden_spoon', 'blue_cup'; ca = env.reg[s][0]; N = len(Q)
def anc(T, a): return T[:, :3, 3] + np.einsum('nij,j->ni', T[:, :3, :3], a)
pa, pt = anc(P[s], ca), anc(tp[s][:N], ca); cc = env.reg[c][0]; qa, qt = anc(P[c], cc), anc(tp[c][:N], cc)
rv = (R.from_matrix(P[s][:, :3, :3]).inv() * R.from_matrix(tp[s][:N, :3, :3])).as_rotvec()   # body-frame rotation achieved->target
# spoon long axis in body frame = principal axis of mesh
import trimesh; v = trimesh.load(sys.argv[2] + '/wooden_spoon.obj', force='mesh').vertices; ev, evec = np.linalg.eigh(np.cov((v - v.mean(0)).T)); ax = evec[:, -1]
print('spoon long axis (body)', ax.round(3), 'extent', (v @ ax).ptp().round(3))
print(' t    pos  deg | dpos world (cm)       | rot body: roll(long) tilt | spoon-cup rel tgt(cm)  ach(cm) | spd tgt deg/s')
wt = R.from_matrix(tp[s][:N, :3, :3]); om = np.degrees(np.linalg.norm((wt[1:] * wt[:-1].inv()).as_rotvec(), axis=1)) * 20
for k0 in range(0, N, 10):
    sl = slice(k0, min(k0 + 10, N)); d = (pt - pa)[sl].mean(0) * 100; r = rv[sl]; roll = np.degrees(np.abs(r @ ax)).mean(); tilt = np.degrees(np.linalg.norm(r - np.outer(r @ ax, ax), axis=1)).mean()
    rt = (pt - qt)[sl].mean(0) * 100; ra = (pa - qa)[sl].mean(0) * 100
    print(f'{k0/20:4.1f} {e[s][0][sl].mean():5.2f} {e[s][1][sl].mean():5.1f} | {d[0]:6.2f} {d[1]:6.2f} {d[2]:6.2f} | {roll:6.1f} {tilt:6.1f} | {rt.round(1)} {ra.round(1)} | {om[sl.start:sl.stop-1].mean() if k0+1<N else 0:5.0f}')
