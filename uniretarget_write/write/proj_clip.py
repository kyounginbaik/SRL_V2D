"""Write a clip whose hand reference is a contact-projected SPIDER-scene trajectory (contact_project.py), with the same object goals.
Usage: proj_clip.py PROJ.npz TASK"""
import sys, numpy as np, mujoco, pyarrow as pa, pyarrow.parquet as pq
from pathlib import Path
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, '/scratch2/exaFLOPs/srl_yam_run/ablate/oak/write')
import wmetric as W, planner as PL
from pool import fk, base, OAK
src, task = sys.argv[1], sys.argv[2]
Q = np.load(src)['q']; n = len(Q); assert n == len(base['robot_right_wrist_position']), (n, len(base['robot_right_wrist_position']))
m = mujoco.MjModel.from_xml_path(str(W.P / 'scene.xml')); d = mujoco.MjData(m); b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'right_hand_C_MC')
jn = list(base['right_robot_finger_joint_names']); ja = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, x)] for x in jn]
names = base['right_robot_frame_names']; wp = np.zeros((n, 3)); wq = np.zeros((n, 4)); fj = Q[:, ja]; F = np.zeros((n, len(names), 7), np.float32)
for t in range(n):
    G = PL.wrist_T(m, d, b, Q[t]); wp[t] = G[:3, 3]; wq[t] = R.from_matrix(G[:3, :3]).as_quat()[[3, 0, 1, 2]]
    Wd = G @ fk(dict(zip(jn, fj[t])), names); F[t, :, :3] = Wd[:, :3, 3]; F[t, :, 3:] = R.from_matrix(Wd[:, :3, :3]).as_quat()[:, [3, 0, 1, 2]]
# the SPIDER wrist frame and the parquet wrist frame must agree: check against the original clip
w0 = np.array(base['robot_right_wrist_position']); Q0 = W.RF['q']; g0 = np.array([PL.wrist_T(m, d, b, Q0[t])[:3, 3] for t in range(0, n, 50)])
print('frame check (original IK vs parquet wrist) cm max', round(float(np.abs(g0 - w0[::50]).max()) * 100, 3))
r = dict(base); r.update(robot_right_wrist_position=wp.tolist(), robot_right_wrist_wxyz=wq.tolist(), robot_right_finger_joints=fj.tolist(), robot_right_frames=F.tolist())
C = OAK / f'clips/{task}'; (C / 'none').mkdir(parents=True, exist_ok=True)
for f in ('object.obj', 'object_rigid.urdf'): (C / f).write_bytes((OAK / f'clips/write_wb/{f}').read_bytes())
r.update(object_mesh_paths=[f'/oak/clips/{task}/object.obj'], object_urdf_paths=[f'/oak/clips/{task}/object_rigid.urdf'])
pq.write_table(pa.Table.from_pylist([r]), C / 'none/reference.parquet'); print('wrote', C)
