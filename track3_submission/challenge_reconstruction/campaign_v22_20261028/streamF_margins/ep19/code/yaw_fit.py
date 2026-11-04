"""Free placement of the settled standing twin: yaw psi about the world vertical through its base contact + table-plane xy shift are physically
free (stable). Choose them so the dust pan reference (target scene anchored on the sweeper frame 0 = scorer alignment) lands on the physical
dust-pan home: minimise |p_ref - p_home| (cm) + w * angle (deg/10). Then settle-check the chosen pose."""
import sys, json, numpy as np, mujoco
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
sys.argv = [sys.argv[0], '0.0005']
exec(open('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamF_margins/ep19/code/twin_rest.py').read().split('res = []')[0])
from common19F import packed_meshes, registration, compose_tracks
reg = registration(packed_meshes()); comp = compose_tracks(); cb = {n: comp[n] @ reg[n]['M'] for n in comp}
Z = np.load('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamF_margins/ep19/twin_poses.npz'); Ts = Z['stable_twin']
mujoco.mj_resetDataKeyframe(m, d, 0); ad = m.joint('mini_dust_pan_free').qposadr[0]
Td = np.eye(4); Td[:3, 3] = d.qpos[ad:ad + 3]; Td[:3, :3] = R.from_quat(d.qpos[ad + 3:ad + 7], scalar_first=True).as_matrix()
an_dp = reg['mini_dust_pan']['candidate_anchor']
W = Pb @ Ts[:3, :3].T + Ts[:3, 3]; c = W[W[:, 2] < W[:, 2].min() + 0.003].mean(0); c[2] = 0
def pose(x):
    psi, dx, dy = x; Rz = R.from_euler('z', psi, degrees=True).as_matrix(); T = np.eye(4); T[:3, :3] = Rz @ Ts[:3, :3]; T[:3, 3] = Rz @ (Ts[:3, 3] - c) + c + [dx, dy, 0]; return T
def err(x):
    S = pose(x); X = S @ np.linalg.inv(cb['mini_sweeper'][0]); Rf = X @ cb['mini_dust_pan'][0]
    pe = np.linalg.norm((Rf[:3, :3] @ an_dp + Rf[:3, 3]) - (Td[:3, :3] @ an_dp + Td[:3, 3])) * 100; ae = np.degrees(R.from_matrix(Rf[:3, :3] @ Td[:3, :3].T).magnitude()); return pe, ae
print('psi=0 no shift', err([0, 0, 0]))
best = None
for psi0 in np.arange(-180, 180, 15):
    r = minimize(lambda x: err(x)[0] + 0.1 * err(x)[1], [psi0, 0, 0], method='Nelder-Mead', options=dict(xatol=1e-4, fatol=1e-5, maxiter=4000))
    if best is None or r.fun < best.fun: best = r
print('best psi %.2f deg shift %.2f %.2f cm' % (best.x[0], best.x[1] * 100, best.x[2] * 100), 'err', err(best.x))
for psi in np.arange(-30, 31, 10):
    r = minimize(lambda x: err([psi, *x])[0], [0, 0], method='Nelder-Mead'); print(' psi', psi, 'fit shift', (r.x * 100).round(2), 'err', np.round(err([psi, *r.x]), 2))
S = pose(best.x); S2, dp, dr = settle(S); print('settle drift %.2f cm %.2f deg' % (dp, dr))
mujoco.mj_resetDataKeyframe(m, d, 0); a_ = m.joint('mini_sweeper_free').qposadr[0]; d.qpos[a_:a_ + 3] = S[:3, 3]; d.qpos[a_ + 3:a_ + 7] = R.from_matrix(S[:3, :3]).as_quat(scalar_first=True); mujoco.mj_forward(m, d)
print('contacts at placement', [(m.geom(d.contact[i].geom1).name, m.geom(d.contact[i].geom2).name, round(d.contact[i].dist * 1000, 2)) for i in range(d.ncon) if 'sweeper' in m.geom(d.contact[i].geom1).name + m.geom(d.contact[i].geom2).name])
np.savez('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamF_margins/ep19/twin_yaw.npz', placed=S, settled=S2, x=best.x)
