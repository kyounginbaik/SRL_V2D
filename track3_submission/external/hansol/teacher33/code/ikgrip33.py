"""T33: is the HUMAN right-hand grip (W1 source Mink IK sharpa_ik.npz, same source frame as its object poses) a firmer spoon grasp?
1) convention check: physics FK of the IK hand (wrist slides = IK pos, rot = IK quat as XYZ euler, fingers by name) mapped to the source frame
   vs the MANO fingertips (hand_targets, hand index 1 = right) -> mean tip distance for quaternion order wxyz vs xyzw.
2) for frames K: put the IK grip on the teacher's EXECUTED spoon (local wrist pose relative to the spoon from the IK file), report per finger
   contact / deepest penetration with the spoon, fingertip + palm positions in the spoon frame (cm), and compare with the teacher's own hand.
usage: ikgrip33.py TEACHER_DIR [frames...]"""
import sys, json, numpy as np, mujoco
from pathlib import Path
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, str(Path(__file__).resolve().parent)); import t33lib as T
from wrist_reference import continuous_euler_xyz
te = Path(sys.argv[1]); K = [int(x) for x in sys.argv[2:]] or [60, 100, 160, 220, 270]
r = json.loads((te / 'result.json').read_text()); m = mujoco.MjModel.from_xml_path(r['execution']['scene']); d = mujoco.MjData(m)
Qx = np.load(r['execution']['trajectory'])['qpos']
man = json.loads((T.W1 / 'input_manifest.json').read_text()); src = Path(man['source']); SfS = np.asarray(man['setup_from_source'])
ik = np.load(src / 'sharpa_ik.npz', allow_pickle=True); RQ = ik['right_qpos']; jn = [str(x) for x in ik['right_joint_names']]; op = np.load(src / 'object_poses.npz'); Wfo = op['world_from_object']; nm = [str(x) for x in op['names']]
si = nm.index('wooden_spoon'); J = np.load(src / 'hand_targets.npz')['hand_joints'].astype(float)
adr = [int(m.joint(f'robot_right_right_{j}').qposadr[0]) for j in ('pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z')]
fcol = [int(m.joint('robot_right_' + n).qposadr[0]) for n in jn]
sa = int(m.joint('wooden_spoon_free').qposadr[0]); bs = m.body('wooden_spoon').id; fm = T.finger_map(m, 'right')
def depth(b):
    n_ = 0
    while b != 0: b = m.body_parentid[b]; n_ += 1
    return n_
tips = {fi: max([b for b, f in fm.items() if f == fi], key=depth) for fi in T.FING}; tips['palm'] = m.body('robot_right_right_hand_C_MC').id
MANO_TIP = {'thumb': 4, 'index': 8, 'middle': 12, 'ring': 16, 'pinky': 20}
def set_hand(q, T_w, fing):
    q = q.copy(); q[adr[:3]] = T_w[:3, 3]; q[adr[3:]] = R.from_matrix(T_w[:3, :3]).as_euler('XYZ'); q[fcol] = fing; return q
def Tq(p, quat, order):
    Tm = np.eye(4); Tm[:3, :3] = R.from_quat(quat, scalar_first=(order == 'wxyz')).as_matrix(); Tm[:3, 3] = p; return Tm
# 1) convention check in the SOURCE frame (physics model, objects irrelevant)
for order in ('wxyz', 'xyzw'):
    errs = []
    for k in K:
        Tw = Tq(RQ[k, :3], RQ[k, 3:7], order); d.qpos[:] = set_hand(Qx[k], Tw, RQ[k, 7:]); mujoco.mj_kinematics(m, d)
        errs.append(np.mean([np.linalg.norm(d.xpos[tips[fi]] - J[k, 1, MANO_TIP[fi]]) for fi in T.FING]))
    print(f'convention {order}: mean fingertip vs MANO tip distance {100*np.mean(errs):.1f} cm (frames {K})')
order = 'wxyz'
for o in ('wxyz', 'xyzw'):
    pass
def report(q, lab):
    d.qpos[:] = q; mujoco.mj_forward(m, d); Rs = d.xmat[bs].reshape(3, 3); ps = d.xpos[bs]; touch = {}; pen = {}
    for i in range(d.ncon):
        c = d.contact[i]; b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
        if bs in (b1, b2):
            o = b2 if b1 == bs else b1
            if o in fm: f = fm[o]; touch[f] = touch.get(f, 0) + 1; pen[f] = min(pen.get(f, 0), c.dist)
    pos = {fi: 100 * Rs.T @ (d.xpos[b] - ps) for fi, b in tips.items()}
    print(f'   {lab:8s} touching: ' + ', '.join(f'{f}({1000*pen[f]:.0f}mm)' for f in sorted(touch, key=(T.FING + ("palm",)).index)) + ' | ' + ' '.join(f'{fi[:2]} {v[0]:5.1f},{v[1]:5.1f},{v[2]:5.1f}' for fi, v in pos.items()))
for order in ('wxyz', 'xyzw'):
    print(f'== IK grip on the executed spoon, quaternion {order} (fingertip/palm positions in the spoon frame, cm; mm = deepest penetration)')
    for k in K:
        Ts_src = Wfo[k, si]; Tw_src = Tq(RQ[k, :3], RQ[k, 3:7], order); local = np.linalg.inv(Ts_src) @ Tw_src
        Ts = np.eye(4); Ts[:3, :3] = R.from_quat(Qx[k, sa + 3:sa + 7], scalar_first=True).as_matrix(); Ts[:3, 3] = Qx[k, sa:sa + 3]
        print(f'  t={k/20:.2f} s  local wrist in spoon frame: p {np.round(100*local[:3,3],1)} cm, rotvec {np.round(np.degrees(R.from_matrix(local[:3,:3]).as_rotvec()),0)} deg')
        report(Qx[k], 'teacher'); report(set_hand(Qx[k], Ts @ local, RQ[k, 7:]), 'IK grip')
