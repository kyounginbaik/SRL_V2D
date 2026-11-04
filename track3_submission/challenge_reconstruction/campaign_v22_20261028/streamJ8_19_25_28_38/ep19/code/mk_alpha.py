"""J8 ep19: sweeper start poses Q_alpha = slerp(stable_twin -> P, alpha) (P = dust-pan-consistent target sweeper f0, topple.py), lowest
collision point at +0.5 mm; saved as sweeper_alpha.npz keys a25/a50/... (input to build_world19J.py --x sw_pose:NPZ:KEY)."""
import sys, numpy as np, mujoco
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamF_margins/ep19/code')
from common19F import *
J = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamJ8_19_25_28_38/ep19')
P = np.load(J / 'sweeper_P.npz')['P']; S = np.load(V / 'twin_poses.npz')['stable_twin']
m = mujoco.MjModel.from_xml_path(str(V / 'world_st/scene/scene_physics_compose_home.xml')); b = m.body('mini_sweeper').id
gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b and m.geom_contype[g] and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]; Pb = []
for g in gs:
    mid = m.geom_dataid[g]; v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]; gm = np.zeros(9); mujoco.mju_quat2Mat(gm, m.geom_quat[g]); Pb.append(v @ gm.reshape(3, 3).T + m.geom_pos[g])
Pb = np.concatenate(Pb); out = {}
for al in (0.25, 0.4, 0.5, 0.6, 0.75, 1.0):
    Rm = R.from_matrix(S[:3, :3]) * R.from_rotvec(al * (R.from_matrix(S[:3, :3]).inv() * R.from_matrix(P[:3, :3])).as_rotvec())
    Q = np.eye(4); Q[:3, :3] = Rm.as_matrix(); Q[:3, 3] = (1 - al) * S[:3, 3] + al * P[:3, 3]; Q[2, 3] += 0.0005 - (Pb @ Q[:3, :3].T + Q[:3, 3])[:, 2].min()
    out['a%02d' % round(al * 100)] = Q
np.savez(J / 'sweeper_alpha.npz', **out); print(list(out))
