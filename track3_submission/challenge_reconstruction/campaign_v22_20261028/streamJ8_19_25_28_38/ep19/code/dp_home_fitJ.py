# J8 ep19 copy of streamF dp_home_fit.py (base dir = J8 ep19)
"""Dust-pan home on the table: keep its physically settled tilt (rest on the table), choose yaw about the vertical through its contact centre + xy
shift (both free on a flat table) to minimise the scorer anchor distance (cm) + 0.05*angle(deg) to the frame-0 target dust pan in WORLD (world_st).
Settle-check; write dp_home.npz (pose + fake 1-frame trajectory for port --pre)."""
import sys, json, numpy as np, mujoco
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
from pathlib import Path
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamF_margins/ep19/code')
from common19F import *
F = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamJ8_19_25_28_38/ep19'); W = sys.argv[1] if len(sys.argv) > 1 else 'world_st'
scn = F / W / 'scene/scene_physics_compose_home.xml'; m = mujoco.MjModel.from_xml_path(str(scn)); d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0)
reg = registration(packed_meshes()); an = reg['mini_dust_pan']['candidate_anchor']
z = np.load(F / W / 'episode_target/episode_000019/reference_trajectory.npz'); on = list(z['object_names']); i = on.index('mini_dust_pan')
Tref = to_T(z['object_positions'][:, i], z['object_quaternions_wxyz'][:, i])
a = m.joint('mini_dust_pan_free').qposadr[0]; Th = np.eye(4); Th[:3, 3] = d.qpos[a:a + 3]; Th[:3, :3] = R.from_quat(d.qpos[a + 3:a + 7], scalar_first=True).as_matrix()
b = m.body('mini_dust_pan').id; gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b and m.geom_contype[g] and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
Pb = []
for g in gs:
    mid = m.geom_dataid[g]; v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]; gm = np.zeros(9); mujoco.mju_quat2Mat(gm, m.geom_quat[g]); Pb.append(v @ gm.reshape(3, 3).T + m.geom_pos[g])
Pb = np.concatenate(Pb); Wp = Pb @ Th[:3, :3].T + Th[:3, 3]; c = Wp[Wp[:, 2] < Wp[:, 2].min() + 0.003].mean(0); c[2] = 0
def pose(x):
    Rz = R.from_euler('z', x[0], degrees=True).as_matrix(); T = np.eye(4); T[:3, :3] = Rz @ Th[:3, :3]; T[:3, 3] = Rz @ (Th[:3, 3] - c) + c + [x[1], x[2], 0]; return T
def err(T, k):
    Rf = Tref[k]; pe = np.linalg.norm((Rf[:3, :3] @ an + Rf[:3, 3]) - (T[:3, :3] @ an + T[:3, 3])) * 100; return pe, np.degrees(R.from_matrix(Rf[:3, :3] @ T[:3, :3].T).magnitude())
ks = list(range(0, 25))   # opening rest frames
f = lambda x: np.mean([err(pose(x), k)[0] + 0.05 * err(pose(x), k)[1] for k in ks])
r = min((minimize(f, [p0, 0, 0], method='Nelder-Mead') for p0 in (-20, -10, 0, 10, 20)), key=lambda r: r.fun)
print('home err f0', np.round(err(Th, 0), 2), '-> fitted yaw %.2f deg shift %.2f %.2f cm' % (r.x[0], r.x[1] * 100, r.x[2] * 100), 'err f0', np.round(err(pose(r.x), 0), 2))
T = pose(r.x); d.qpos[a:a + 3] = T[:3, 3]; d.qpos[a + 3:a + 7] = R.from_matrix(T[:3, :3]).as_quat(scalar_first=True); d.qvel[:] = 0
if m.key_ctrl.size: d.ctrl[:] = m.key_ctrl[0]
mujoco.mj_forward(m, d); con = [(m.geom(d.contact[j].geom1).name, m.geom(d.contact[j].geom2).name, round(d.contact[j].dist * 1000, 1)) for j in range(d.ncon)]
print('non-table contacts of dust pan', [x for x in con if 'dust' in x[0] + x[1] and 'table' not in x[0] + x[1]])
for _ in range(1000): mujoco.mj_step(m, d)
T2 = np.eye(4); T2[:3, 3] = d.qpos[a:a + 3]; T2[:3, :3] = R.from_quat(d.qpos[a + 3:a + 7], scalar_first=True).as_matrix()
print('settle drift %.3f cm %.2f deg' % (np.linalg.norm(T2[:3, 3] - T[:3, 3]) * 100, np.degrees(R.from_matrix(T2[:3, :3] @ T[:3, :3].T).magnitude())), 'err f0 settled', np.round(err(T2, 0), 2))
q = R.from_matrix(T2[:3, :3]).as_quat(scalar_first=True)
np.savez(F / f'dp_home_{W}.npz', placed=T, settled=T2, x=r.x, object_names=np.array(['mini_dust_pan']), object_positions=T2[None, None, :3, 3], object_quaternions_wxyz=q[None, None])
