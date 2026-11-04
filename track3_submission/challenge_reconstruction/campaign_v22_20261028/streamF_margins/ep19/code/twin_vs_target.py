import sys, json, numpy as np, mujoco
from scipy.spatial.transform import Rotation as R
sys.argv = [sys.argv[0], '0.0005']
exec(open('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamF_margins/ep19/code/twin_rest.py').read().split('res = []')[0])
from common19F import packed_meshes, registration, compose_tracks
reg = registration(packed_meshes()); comp = compose_tracks(); cb = {n: comp[n] @ reg[n]['M'] for n in comp}
mujoco.mj_resetDataKeyframe(m, d, 0); ad = m.joint('mini_dust_pan_free').qposadr[0]
Td = np.eye(4); Td[:3, 3] = d.qpos[ad:ad + 3]; Td[:3, :3] = R.from_quat(d.qpos[ad + 3:ad + 7], scalar_first=True).as_matrix()
Xdp = Td @ np.linalg.inv(cb['mini_dust_pan'][0]); Tsw_t = Xdp @ cb['mini_sweeper'][0]
def cmp(A, B, s): print(s, '%.2f cm %.1f deg' % (np.linalg.norm(A[:3, 3] - B[:3, 3]) * 100, np.degrees(R.from_matrix(A[:3, :3] @ B[:3, :3].T).magnitude())))
cmp(Tsw_t, Tk, 'target sweeper f0 (dust-pan anchored) vs our home'); cmp(Tsw_t, Tw, '... vs exact twin of our home')
Ts, dp, dr = settle(place(8, -8)); cmp(Ts, Tw, 'stable twin vs exact twin'); cmp(Tsw_t, Ts, 'target sweeper f0 vs stable twin')
W = Pb @ Tsw_t[:3, :3].T + Tsw_t[:3, 3]; print('target sweeper f0 lowest z (dp anchored) cm %.2f' % (W[:, 2].min() * 100), 'body z world', Tsw_t[:3, 2].round(3), 'stable twin body z', Ts[:3, 2].round(3), 'home z', Tk[:3, 2].round(3))
Ts2, dp2, dr2 = settle(Tsw_t + np.diag([0, 0, 0, 0]) + np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0.0+ (0.0005 - W[:, 2].min())],[0,0,0,0]])); cmp(Ts2, Tsw_t, 'target sweeper f0 (lifted to clear) settle drift')
np.savez(str(Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamF_margins/ep19/twin_poses.npz')), home=Tk, twin=Tw, stable_twin=Ts, target_f0_dp=Tsw_t, target_f0_dp_settled=Ts2, Xdp=Xdp)
