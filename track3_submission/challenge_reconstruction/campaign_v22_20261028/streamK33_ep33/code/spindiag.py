"""K33-HG spin diagnosis: per-frame spoon rate/roll, right-hand contacts, wrist tracking. usage: spindiag.py VAR(HGA) RUN(c3) F0 F1"""
import sys, json, numpy as np, mujoco
from pathlib import Path
from scipy.spatial.transform import Rotation as R
V, N, F0, F1 = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
C = Path(f'/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates/K33_K33_cw{V}fix_{V}_{N}/episode_000033/backend/evaluation/execution')
z = np.load(C / 'trajectory.npz', allow_pickle=False); m = mujoco.MjModel.from_xml_path(str(C / 'scene_physics.xml')); d = mujoco.MjData(m)
ob = [str(x) for x in z['object_names']]; k = ob.index('wooden_spoon'); q = z['object_quaternions_wxyz'][:, k]; Rs = R.from_quat(q[:, [1, 2, 3, 0]])
sim_rate = np.degrees((Rs[:-1].inv() * Rs[1:]).magnitude())
body_rotvec = np.degrees((Rs[:-1].inv() * Rs[1:]).as_rotvec())
cz = np.load(f'/data1/SRL_V2D/challenge_submissions_SRL/cowork_ep33_recon_HGA_B_20261102/x_{V}/recon_E14_best_B2A_S1_{V}_sharpa/episode_000033/object_poses.npz', allow_pickle=False)
n = [str(x) for x in cz['names']]; Tt = cz['world_from_object'][:, n.index('wooden_spoon')]; Rt = R.from_matrix(Tt[:, :3, :3]); tg_rate = np.degrees((Rt[:-1].inv() * Rt[1:]).magnitude())
seg = np.load(f'/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK33_ep33/cw/out/seg_{V}{N}.npz', allow_pickle=False); err = seg[f'{N}_sp_deg']; roll = seg[f'{N}_sp_roll']
jn = [str(x) for x in z['joint_names']]; an = [str(x) for x in z['actuator_names']]
print('joints', jn[:8], '...', 'actuators', an[:8]); print('forces shape', z['hand_object_force_n'].shape)
gname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ''; bname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ''
sp_body = [i for i in range(m.nbody) if 'spoon' in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or '')]; print('spoon bodies', [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in sp_body])
qv = z['qvel']; qp = z['qpos']
for f in range(F0, F1):
    d.qpos[:] = qp[f]; d.qvel[:] = qv[f]; d.ctrl[:] = z['ctrl'][f]; mujoco.mj_forward(m, d)
    cs = {}
    for c in range(d.ncon):
        g1, g2 = d.contact[c].geom1, d.contact[c].geom2; b1, b2 = m.geom_bodyid[g1], m.geom_bodyid[g2]
        if b1 in sp_body or b2 in sp_body:
            o = b2 if b1 in sp_body else b1; nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, o) or 'world'
            f6 = np.zeros(6); mujoco.mj_contactForce(m, d, c, f6); cs[nm] = cs.get(nm, 0) + abs(f6[0])
    print(f'f{f} t{f/20:.2f} err {err[f]:5.0f} roll {roll[f]:+5.0f} simrate {sim_rate[f]:5.1f} tgrate {tg_rate[f]:4.1f} F {z["hand_object_force_n"][f,0]:.1f},{z["hand_object_force_n"][f,1]:.1f} track {z["track_pos_m"][f]*100:.1f}/{z["track_rot_deg"][f]:.0f} wrF {z["wrist_force_n"][f]:.0f} | ' + ' '.join(f'{a.replace("right_","r_").replace("left_","l_")}:{b:.1f}' for a, b in sorted(cs.items())))
