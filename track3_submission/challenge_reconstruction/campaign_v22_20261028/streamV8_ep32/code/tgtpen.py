"""V8: does the TARGET spoon (target_final_v5 in the physics world) intersect the target cup / sit inside it? per 0.25 s:
spoon tip (both long-axis ends) in the cup frame (radial, height above cup bottom), mujoco spoon-cup contacts and min distance with both objects at target poses."""
import os, sys, numpy as np, mujoco
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamM8_32_33_15_16/code')
import mres
env = mres.Env(dict(ep=32, actor=sys.argv[1], meshes=sys.argv[2])); Q = np.load(sys.argv[3])['Q']
_, A = mres.L.per_frame(32, env.poses(Q), env.reg, env.tgt, want_A=True); tp = mres.L.target_phys(32, env.reg, A, env.tgt)
m = env.m; d = mujoco.MjData(m); sa, ca = env.oa['wooden_spoon'], env.oa['blue_cup']; sp = m.body('wooden_spoon').id; cu = m.body('blue_cup').id
import trimesh
from scipy.spatial.transform import Rotation as R
vs = trimesh.load(sys.argv[2] + '/wooden_spoon.obj', force='mesh').vertices; vc = trimesh.load(sys.argv[2] + '/blue_cup.obj', force='mesh').vertices
print('cup body-frame bbox', vc.min(0).round(3), vc.max(0).round(3)); i0, i1 = np.argmin(vs[:, 0]), np.argmax(vs[:, 0]); ends = vs[[i0, i1]]
print('spoon ends (body)', ends.round(3))
for k in range(0, env.T, 5):
    d.qpos[:] = Q[k]
    for a, T in ((sa, tp['wooden_spoon'][k]), (ca, tp['blue_cup'][k])):
        d.qpos[a:a + 3] = T[:3, 3]; d.qpos[a + 3:a + 7] = R.from_matrix(T[:3, :3]).as_quat(scalar_first=True)
    mujoco.mj_forward(m, d); pen = [d.contact[i].dist for i in range(d.ncon) if {m.geom_bodyid[d.contact[i].geom1], m.geom_bodyid[d.contact[i].geom2]} == {sp, cu}]
    Tc = tp['blue_cup'][k]; Ts = tp['wooden_spoon'][k]; e = (np.linalg.inv(Tc) @ (Ts @ np.c_[ends, np.ones(2)].T)).T[:, :3]
    # achieved too
    Ec = env.poses(Q[k:k+1]); ea = (np.linalg.inv(Ec['blue_cup'][0]) @ (Ec['wooden_spoon'][0] @ np.c_[ends, np.ones(2)].T)).T[:, :3]
    f = lambda p: f'r={np.hypot(p[0], p[1])*100:5.1f} h={p[2]*100:5.1f}'
    print(f'{k/20:5.2f} tgt end0 {f(e[0])} end1 {f(e[1])} | ach end0 {f(ea[0])} end1 {f(ea[1])} | tgt spoon-cup contacts {len(pen)} min {min(pen)*1000 if pen else 0:6.1f} mm')
