import os, sys, json, numpy as np, mujoco
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import planC
from scipy.spatial.transform import Rotation as R
pl = planC.Plan(sys.argv[1]); m = mujoco.MjModel.from_xml_path(pl.scene); d = mujoco.MjData(m)
ev = json.load(open(sys.argv[1] + '/evaluation/result.json')); tr = np.load(ev['execution']['trajectory'])
jn = ['robot_right_right_rot_x', 'robot_right_right_rot_y', 'robot_right_right_rot_z']
qa = [m.jnt_qposadr[m.joint(j).id] for j in jn]; print([m.joint(j).type for j in jn], [m.jnt_axis[m.joint(j).id] for j in jn])
pid = m.body('robot_right_right_hand_C_MC').id; sid = m.body('wooden_spoon').id
off = None
for f in range(0, pl.n, 40):
    d.qpos[:] = tr['qpos'][f]; mujoco.mj_kinematics(m, d)
    Rp = R.from_quat(d.xquat[pid][[1, 2, 3, 0]]); Re = R.from_euler('XYZ', tr['qpos'][f][qa]); Rx = R.from_euler('xyz', tr['qpos'][f][qa])
    if off is None: off = Re.inv() * Rp; offx = Rx.inv() * Rp
    Rs = R.from_quat(d.xquat[sid][[1, 2, 3, 0]])
    print(f, 'XYZ intrinsic err', round(np.degrees(((Re * off).inv() * Rp).magnitude()), 2), 'xyz extrinsic err', round(np.degrees(((Rx * offx).inv() * Rp).magnitude()), 2),
          'spoon traj vs xquat', round(np.degrees((pl.Re[f].inv() * Rs).magnitude()), 2), 'ctrl vs qpos', np.round(pl.sc[np.searchsorted(pl.st, f / 20)][pl.ids[3:]] - tr['qpos'][f][qa], 3))
