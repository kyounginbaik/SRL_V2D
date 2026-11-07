"""Official Do As I Do stages for write_wb, in launch.py's order and settings. One addition: after the pedestal stage, the same board goes into scene.xml and scene_eq.xml, placed by the rigid map from our scene to theirs, fitted on the pen's own trajectory."""
import sys, json, numpy as np, re
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import launch as L
from launch import *
H = Path('/scratch2/exaFLOPs/srl_yam_run/ablate/oak/write'); B = json.load(open(H / 'board.json')); T = 'write_wb'; RAW = '/scratch2/exaFLOPs/daid_oakink/raw/write_wb'
D = dict(dataset_name='do_as_i_do', data_id=0)
task = L.process_dataset(raw_dir=RAW, output_root_dir='outputs', task=T, data_id=0, embodiment_type='right', dataset_name='do_as_i_do', force=True); assert task
L.decompose_mesh(task=task, dataset_name='do_as_i_do', data_id=0, embodiment_type='right', thicken=.002, dilate=.002, force=True)
L.generate_scene(task=task, dataset_name='do_as_i_do', data_id=0, embodiment_type='right', robot_type='sharpa', show_viewer=False, friction_scale=1.5, object_floor_collision=L.OBJECT_FLOOR_COLLISION, hand_floor_collision=L.HAND_FLOOR_COLLISION, use_pedestal=L.USE_PEDESTAL, use_support=L.USE_SUPPORT, force=True, add_ur3_arm=False)
L.solve_ik(task=task, dataset_name='do_as_i_do', data_id=0, embodiment_type='right', robot_type='sharpa', show_viewer=False, force=True, smoothing=True)
cfg = L.load_mjwp_config(dataset_name='do_as_i_do', task=task, data_id=0, robot_type='sharpa', embodiment_type='right', seed=0, wait_on_finish=False, max_sim_steps=0, force=True, show_viewer=False)
L.resolve_scene_pedestal(output_root_dir='outputs', dataset_name='do_as_i_do', robot_type='sharpa', embodiment_type='right', task=task, data_id=0, use_pedestal=L.USE_PEDESTAL, use_support=L.USE_SUPPORT, hand_object_distance_thresh=cfg.hand_object_distance_thresh, force=True)
O = Path(f'outputs/sharpa/right/{task}'); print('scene dir', O, list(O.iterdir()))
import mujoco, xml.etree.ElementTree as ET
kin = next(O.rglob('trajectory_kinematic.npz')); scene = kin.parent / 'scene.xml'; m = mujoco.MjModel.from_xml_path(str(scene)); a = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'right_object_joint')]
k = np.load(kin)['qpos']; k = k.reshape(-1, m.nq); pen = k[:, a:a + 7]; print('DAID ref', k.shape)
ours = np.load(H / 'daid_pen_ours.npy')  # pen in our scene at their frames, written by the adapter step
n = min(len(pen), len(ours)); pen, ours = pen[:n], ours[:n]
Ra = R.from_quat(pen[:, 3:][:, [1, 2, 3, 0]]).as_matrix() @ R.from_quat(ours[:, 3:][:, [1, 2, 3, 0]]).as_matrix().transpose(0, 2, 1); U, _, V = np.linalg.svd(Ra.mean(0)); A = U @ V; t = (pen[:, :3] - ours[:, :3] @ A.T).mean(0); res = np.linalg.norm(ours[:, :3] @ A.T + t - pen[:, :3], axis=1); print('map', np.round(A, 3).tolist(), t.round(4), 'residual cm mean', res.mean() * 100, 'max', res.max() * 100); assert res.max() < .02
pos = A @ np.array(B['pos']) + t; q = (R.from_matrix(A) * R.from_quat(np.array(B['quat'])[[1, 2, 3, 0]])).as_quat()[[3, 0, 1, 2]]
json.dump(dict(A=A.tolist(), t=t.tolist(), pos=pos.tolist(), quat=q.tolist()), open(H / 'board_daid.json', 'w'))
for f in ('scene.xml', 'scene_eq.xml'):
    p = kin.parent / f
    if not p.exists(): p = O / f
    if not p.exists(): print('skip', f); continue
    s = p.read_text()
    if 'name="board"' in s: continue
    g = [e.get('name') for e in ET.fromstring(s).iter('geom') if (e.get('name') or '').startswith(('right_object_', 'collision_hand_')) and e.get('name') != 'right_object_visual']
    geom = f'<geom name="board" type="box" size="{B["half"][0]} {B["half"][1]} {B["half"][2]}" pos="{pos[0]:.5f} {pos[1]:.5f} {pos[2]:.5f}" quat="{q[0]:.5f} {q[1]:.5f} {q[2]:.5f} {q[3]:.5f}" contype="0" conaffinity="0" rgba="0.92 0.92 0.88 1"/>\n'
    s = re.sub(r'(<worldbody>\s*)', lambda M: M.group(1) + geom, s, count=1)
    pairs = ''.join(f'<pair geom1="board" geom2="{x}" name="board_{x}" friction="{".3 .3 .1 0 0" if x.startswith("right_object") else "1 1 .1 0 0"}"/>\n' for x in g)
    assert '</contact>' in s; p.write_text(s.replace('</contact>', pairs + '</contact>', 1)); print('board added', p, len(g))
L.optimize_physics(cfg); print('DAID_DONE')
