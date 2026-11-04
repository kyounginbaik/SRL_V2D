"""Stream R ep32: from a hold-to-end rollout (evaluation/result.json of a rewrap32*_v2 'hold' variant), measure the constant left-wrist place rotation
(local rotvec, rad) and compensating translation (m) that carry the achieved cup axis onto the reference cup axis over frames F0..end
(reference = the evaluation's input_dir source), rotation about the left wrist base. Prints the rewrap32_v2 arguments for scale 1.
usage: measure_hold_rotation.py EVAL_RESULT_JSON [F0=222]"""
import sys, json, numpy as np, mujoco, trimesh
from scipy.spatial.transform import Rotation as R
ev = json.load(open(sys.argv[1])); f0 = int(sys.argv[2]) if len(sys.argv) > 2 else 222
tr = np.load(ev['execution']['trajectory']); man = json.load(open(ev['input_dir'] + '/input_manifest.json')); Sm = np.asarray(man['setup_from_source'])
z = np.load(man['source'] + '/object_poses.npz'); W = z['world_from_object'][:, list(z['names']).index('blue_cup')]
anc = trimesh.load('/data1/SRL_V2D/challenge_reconstruction/final_recon/episode_000032/assets/blue_cup.obj', force='mesh', process=False).centroid
m = mujoco.MjModel.from_xml_path(ev['execution']['scene']); d = mujoco.MjData(m)
an = [str(x) for x in tr['actuator_names']]; ri = [an.index(f'robot_left_left_rot_{a}_position') for a in 'xyz']; ci = [str(x) for x in tr['object_names']].index('blue_cup')
ws, dps, vl = [], [], []
for k in range(f0, len(tr['qpos'])):
    Ra = R.from_quat(tr['object_quaternions_wxyz'][k, ci][[1, 2, 3, 0]]).as_matrix(); pa = tr['object_positions'][k, ci] + Ra @ anc
    Tr = Sm @ W[k]; pr = Tr[:3, :3] @ anc + Tr[:3, 3]; aa, ar = Ra[:, 2], Tr[:3, 2]
    ang = np.arccos(np.clip(aa @ ar, -1, 1)); ax = np.cross(aa, ar); ax /= np.linalg.norm(ax); w = ax * ang
    d.qpos[:] = tr['qpos'][k]; mujoco.mj_kinematics(m, d); pw = d.body('robot_left_left_base_yaw').xpos.copy()
    Rw = R.from_rotvec(w).as_matrix(); dps.append((pr - pa) - (Rw - np.eye(3)) @ (pa - pw)); ws.append(w); vl.append(R.from_euler('XYZ', tr['ctrl'][k, ri]).as_matrix().T @ w)
w, v, t = np.mean(ws, 0), np.mean(vl, 0), np.mean(dps, 0)
print(json.dumps(dict(world_rotvec_deg=np.degrees(w).round(2).tolist(), angle_deg=float(np.degrees(np.linalg.norm(w))), spread_deg=np.degrees(np.std(ws, 0)).round(2).tolist(), local_rotvec=v.round(5).tolist(), translation_m=t.round(5).tolist())))
print('ARGS rot_x=%.5f rot_y=%.5f rot_z=%.5f place_dx=%.5f place_dy=%.5f place_dz=%.5f' % (*v, *t))
