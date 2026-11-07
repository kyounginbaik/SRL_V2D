"""Pool of physically checked grips: reruns planner grips under official fingers, keeps the settled finger pose and hand-pen relation, and writes each as a reference (clips/write_gN). The hand follows the true pen path rigidly, so every reset frame starts in that grip. Object goals are unchanged. Also writes low-overlap checks."""
import os, sys, json, numpy as np, mujoco, pyarrow as pa, pyarrow.parquet as pq, multiprocessing as mp
os.environ['REAL'] = '1'; os.environ['OPEN'] = '.2'
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import planner as PL, wmetric as W
OAK = Path('/scratch2/exaFLOPs/srl_yam_run/ablate/oak'); XMLD = '/scratch2/exaFLOPs/SRL_V2D/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/xmls/sharpawave'
base = pq.read_table(OAK / 'clips/write_wb/none/reference.parquet').to_pylist()[0]
hand = mujoco.MjModel.from_xml_path(f'{XMLD}/right_sharpawave.xml'); hd = mujoco.MjData(hand)
def fk(q, names):
    hd.qpos[:] = 0
    for k, v in q.items(): hd.qpos[hand.jnt_qposadr[mujoco.mj_name2id(hand, mujoco.mjtObj.mjOBJ_JOINT, k)]] = v
    mujoco.mj_kinematics(hand, hd); root = mujoco.mj_name2id(hand, mujoco.mjtObj.mjOBJ_BODY, 'right_hand_C_MC'); G = np.eye(4); G[:3, :3] = hd.xmat[root].reshape(3, 3); G[:3, 3] = hd.xpos[root]; Gi = np.linalg.inv(G); out = []
    for nm in names:
        M = np.eye(4); s_ = mujoco.mj_name2id(hand, mujoco.mjtObj.mjOBJ_SITE, nm); b_ = mujoco.mj_name2id(hand, mujoco.mjtObj.mjOBJ_BODY, nm); j_ = mujoco.mj_name2id(hand, mujoco.mjtObj.mjOBJ_JOINT, nm)
        if s_ >= 0: M[:3, :3] = hd.site_xmat[s_].reshape(3, 3); M[:3, 3] = hd.site_xpos[s_]
        elif b_ >= 0: M[:3, :3] = hd.xmat[b_].reshape(3, 3); M[:3, 3] = hd.xpos[b_]
        elif j_ >= 0: bid = hand.jnt_bodyid[j_]; M[:3, :3] = hd.xmat[bid].reshape(3, 3); M[:3, 3] = hd.xanchor[j_]
        elif nm in ('universe', 'root_joint'): M = G.copy()
        else: raise KeyError(nm)
        out.append(Gi @ M)
    return np.stack(out)
def build(i, seed_row):
    os.environ['REC'] = str(W.H / f'pool_{i}.npz'); x = PL.rollout(tuple(seed_row)); assert x['PASS'], x
    z = np.load(os.environ['REC']); Xh = z['Xh']; qs = z['q_settled']; m = mujoco.MjModel.from_xml_path(str(W.P / 'scene.xml')); jn = list(base['right_robot_finger_joint_names']); fj = np.array([qs[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]] for n in jn])
    # Same goals as the reference it replaces, frame for frame.
    op = np.array(base['object_root_position']); oq = R.from_rotvec(np.array(base['object_root_axis_angle'])); n = len(op); Xi = np.linalg.inv(Xh); wr = oq * R.from_matrix(Xi[:3, :3]); wp = op + oq.apply(Xi[:3, 3]); wq = wr.as_quat()[:, [3, 0, 1, 2]]
    names = base['right_robot_frame_names']; Q = fk(dict(zip(jn, fj)), names); F = np.zeros((n, len(names), 7), np.float32)
    for t in range(n):
        G = np.eye(4); G[:3, :3] = wr[t].as_matrix(); G[:3, 3] = wp[t]; Wd = G @ Q; F[t, :, :3] = Wd[:, :3, 3]; q_ = R.from_matrix(Wd[:, :3, :3]).as_quat(); F[t, :, 3:] = q_[:, [3, 0, 1, 2]]
    r = dict(base); r.update(robot_right_wrist_position=wp.tolist(), robot_right_wrist_wxyz=wq.tolist(), robot_right_finger_joints=np.tile(fj, (n, 1)).tolist(), robot_right_frames=F.tolist())
    C = OAK / f'clips/write_g{i}'; (C / 'none').mkdir(parents=True, exist_ok=True)
    for f in ('object.obj', 'object_rigid.urdf'): (C / f).write_bytes((OAK / f'clips/write_wb/{f}').read_bytes())
    r.update(object_mesh_paths=[f'/oak/clips/write_g{i}/object.obj'], object_urdf_paths=[f'/oak/clips/write_g{i}/object_rigid.urdf']); pq.write_table(pa.Table.from_pylist([r]), C / 'none/reference.parquet')
    return dict(i=i, seed=seed_row[-1], tip=x['tip_med_cm'], overlap=x['pen_depth_cm'], hold=x['hold_cm'])
R_ = [json.loads(l) for l in open(W.H / 'planner_real.jsonl')]; P = {r['seed']: r for r in R_ if r['PASS'] and r.get('pen_depth_cm', 9) <= .3}; rob = json.load(open(W.H / 'robust_real.json')); cnt = {}
for x in rob: cnt[x['seed']] = cnt.get(x['seed'], 0) + bool(x['PASS'])
top = sorted(P, key=lambda s: (-cnt.get(s, 0), P[s]['tip_med_cm']))[:6]; print('top seeds', [(s, cnt.get(s)) for s in top])
out = []
for i, s in enumerate(top):
    r = P[s]; out.append(build(i, (r['f'], r['squeeze'], r['servo'], r['press'], s))); print(out[-1], flush=True)
json.dump(out, open(W.H / 'pool.json', 'w'))
