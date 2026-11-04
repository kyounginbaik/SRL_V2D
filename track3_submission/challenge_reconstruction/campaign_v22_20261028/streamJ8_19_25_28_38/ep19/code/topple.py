"""J8 ep19: sweeper start pose consistent with the dust pan (P = dp_settled @ inv(target dp f0) @ target sweeper f0, world_st) -
lowest point and passive topple timeline with the robot held at its keyframe."""
import sys, numpy as np, mujoco
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamF_margins/ep19/code')
from common19F import *
z = np.load(V / 'world_st/episode_target/episode_000019/reference_trajectory.npz'); on = list(z['object_names'])
T = {n: to_T(z['object_positions'][:, on.index(n)], z['object_quaternions_wxyz'][:, on.index(n)]) for n in NAMES}
st = np.load(V / 'dp_home_world_st.npz')['settled']; tw = np.load(V / 'twin_poses.npz')
P = st @ np.linalg.inv(T['mini_dust_pan'][0]) @ T['mini_sweeper'][0]
m = mujoco.MjModel.from_xml_path(str(V / 'world_st/scene/scene_physics_compose_home.xml')); d = mujoco.MjData(m)
a = m.joint('mini_sweeper_free').qposadr[0]; b = m.body('mini_sweeper').id
gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b and m.geom_contype[g] and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
Pb = []
for g in gs:
    mid = m.geom_dataid[g]; v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]; gm = np.zeros(9); mujoco.mju_quat2Mat(gm, m.geom_quat[g]); Pb.append(v @ gm.reshape(3, 3).T + m.geom_pos[g])
Pb = np.concatenate(Pb)
low = lambda Tm: (Pb @ Tm[:3, :3].T + Tm[:3, 3])[:, 2].min()
for k in ('stable_twin', 'twin'): print(k, 'collision lowz cm %.2f' % (low(tw[k]) * 100), 'angle to P %.1f' % np.degrees(R.from_matrix(tw[k][:3, :3].T @ P[:3, :3]).magnitude()))
print('P collision lowz cm %.2f' % (low(P) * 100), 'P vs stable_twin pos cm %.2f' % (np.linalg.norm(P[:3, 3] - tw['stable_twin'][:3, 3]) * 100))
com = d.xipos if False else None
for alpha in [float(x) for x in (sys.argv[1:] or ['1.0', '0.75', '0.5', '0.25'])]:
    S = tw['stable_twin']; Rm = R.from_matrix(S[:3, :3]) * R.from_rotvec(alpha * (R.from_matrix(S[:3, :3]).inv() * R.from_matrix(P[:3, :3])).as_rotvec())
    Q = np.eye(4); Q[:3, :3] = Rm.as_matrix(); Q[:3, 3] = (1 - alpha) * S[:3, 3] + alpha * P[:3, 3]; Q[2, 3] += 0.0005 - low(Q)
    mujoco.mj_resetDataKeyframe(m, d, 0); d.qpos[a:a + 3] = Q[:3, 3]; d.qpos[a + 3:a + 7] = R.from_matrix(Q[:3, :3]).as_quat(scalar_first=True); d.qvel[:] = 0
    if m.key_ctrl.size: d.ctrl[:] = m.key_ctrl[0]
    mujoco.mj_forward(m, d); out = []
    for k in range(int(1.5 / m.opt.timestep)):
        mujoco.mj_step(m, d)
        if k % int(0.25 / m.opt.timestep) == 0:
            Rt = R.from_quat(d.qpos[a + 3:a + 7], scalar_first=True).as_matrix(); out.append('%.2fs %.1fdeg %.2fcm' % (d.time, np.degrees(R.from_matrix(Rt.T @ Q[:3, :3]).magnitude()), np.linalg.norm(d.qpos[a:a + 3] - Q[:3, 3]) * 100))
    print('alpha', alpha, ' | '.join(out))
np.savez('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamJ8_19_25_28_38/ep19/sweeper_P.npz', P=P)
