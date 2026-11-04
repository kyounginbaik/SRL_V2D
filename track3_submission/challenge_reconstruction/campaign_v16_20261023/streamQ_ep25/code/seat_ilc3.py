"""Stream Q ep25: insertion re-plan by a constant right-wrist ROTATION correction (one ILC step). In the executed source rollout,
board 2 approaches board 1 at 62-73 deg between the long axes and tilted 13-29 deg (seated cross: 90 / 0), which is why it never
engages the slot. Correction Rc = mean over [T1, T1+DUR] of R_des R_act^T, R_des = (actual board 1) x (assembly rotation of the
two boards: long axes perpendicular, mid axes parallel, signs of the reference's final cross); applied with GAIN to the right wrist
command about board 2's centre (wrist pivot moved accordingly), weight smoothstep T0->T1 then 1 until the end. Fingers unchanged.
The corrected commands are executed in the world scene and scored against the world's corrected reference.
v2 (seat_ilc2.py): also a constant TRANSLATION correction (TGAIN) moving board 2's slot centre onto board 1's plate mid-plane and
board 2's plate onto board 1's slot centre (both along board 1's thin / long axes, mean over the same window); vertical untouched.
v3 (seat_ilc3.py): plus a vertical clearance profile on the right wrist: +LIFT m ramped in over [L0-0.5, L0], held until L1, ramped
out (descent into the slot) by L2, so board 2 is aligned above board 1 before it engages.
usage: seat_ilc3.py SRC_DIR WORLD_DIR OUT_DIR T0 T1 DUR RGAIN TGAIN LIFT L0 L1 L2 [SCENE]"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import sys, json
from pathlib import Path
import numpy as np, mujoco
from scipy.spatial.transform import Rotation as R
LEG = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017')
sys.path[:0] = [str(LEG / 'code'), '/data1/SRL_V2D/third_party/DexAgent', str(Path(__file__).resolve().parent)]
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files
from wrist_reference import continuous_euler_xyz
import correct_ep25_v2 as C
src, world, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve(); T0, T1, DUR, GAIN, TGAIN, LIFT, L0, L1, L2 = map(float, sys.argv[4:13])
scene = Path(sys.argv[13]) if len(sys.argv) > 13 else world / 'scene/scene_physics_corrected.xml'; epdir = world / 'episodes/episode_000025'
man = json.loads((epdir / 'input_manifest.json').read_text()); out.mkdir(parents=True, exist_ok=False)
F = C.frames(Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/collision_recon_slots_v3/episode_000025')); f1, f2 = F['wooden_piece_1'], F['wooden_piece_2']
z = {k: v for k, v in np.load(src / 'reference.npz', allow_pickle=False).items()}; Q = np.asarray(z['qpos'], float).copy(); names = list(np.asarray(z['joint_names']).astype(str))
tr = np.load(json.loads((src / 'result.json').read_text())['execution']['trajectory'])
def pose(k, j): M = np.eye(4); M[:3, :3] = R.from_quat(tr['object_quaternions_wxyz'][k, j][[1, 2, 3, 0]]).as_matrix(); M[:3, 3] = tr['object_positions'][k, j]; return M
ref0 = np.load(world / 'source_corrected/episode_000025/object_poses.npz')['world_from_object']
_, P = C.assembly(F, ref0[350, 0], ref0[350, 1]); Xr = f1['U'] @ P @ f2['U'].T
k1, k2 = int(T1 * 20), int((T1 + DUR) * 20); rots = []; dts = []
for k in range(k1, k2 + 1):
    T1w, T2w = pose(k, 0), pose(k, 1); rots.append(R.from_matrix((T1w[:3, :3] @ Xr) @ T2w[:3, :3].T))
    s2 = f2['c0'] + f2['U'] @ np.array([f2['plate_thin'], 0, f2['slot_long']]); p = T2w[:3, :3] @ s2 + T2w[:3, 3]
    q = f1['U'].T @ (T1w[:3, :3].T @ (p - T1w[:3, 3]) - f1['c0']); c2 = T2w[:3, :3] @ f2['c0'] + T2w[:3, 3]; qc = f1['U'].T @ (T1w[:3, :3].T @ (c2 - T1w[:3, 3]) - f1['c0'])
    dts.append(T1w[:3, :3] @ f1['U'] @ np.array([f1['plate_thin'] - q[0], 0., f1['slot_long'] - qc[2]]))
Rc_full = R.concatenate(rots).mean(); rv = Rc_full.as_rotvec() * GAIN; dtv = np.mean(dts, 0) * TGAIN; dtv[2] = 0.
c2_body = f2['slab_c']
m = mujoco.MjModel.from_xml_path(str(scene)); d = mujoco.MjData(m)
cols = [names.index(f'robot_right_right_{a}') for a in ('pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z')]

assert names == [n for n in names]
from tool_library.common import episode as E
assert names == E.joint_names(m)
newQ = Q.copy()
for k in range(len(Q)):
    t = k / 20.
    sm = lambda u: float(np.clip(u, 0, 1)) ** 2 * (3 - 2 * float(np.clip(u, 0, 1)))
    lz = LIFT * (sm((t - (L0 - .5)) / .5) if t < L1 else 1 - sm((t - L1) / (L2 - L1)))
    newQ[k, cols[2]] = Q[k, cols[2]] + lz
    if t < T0: continue
    w = 1. if t >= T1 else (lambda u: u * u * (3 - 2 * u))((t - T0) / (T1 - T0)); Rk = R.from_rotvec(w * rv)
    kk = min(k, len(tr['object_positions']) - 1); T2w = pose(kk, 1); c2 = T2w[:3, :3] @ c2_body + T2w[:3, 3]
    d.qpos[:] = Q[k]; mujoco.mj_kinematics(m, d); piv = d.xanchor[m.joint('robot_right_right_rot_x').id].copy()
    Rw = R.from_euler('XYZ', Q[k, cols[3:6]]); Rn = Rk * Rw
    newpiv = c2 + Rk.apply(piv - c2); newQ[k, cols[:3]] = Q[k, cols[:3]] + (newpiv - piv) + w * dtv; newQ[k, cols[2]] += lz
    newQ[k, cols[3:6]] = continuous_euler_xyz(Rn, initial_angles=Q[k, cols[3:6]]) if False else Rn.as_euler('XYZ')
# unwrap euler continuity
for c in cols[3:6]: newQ[:, c] = np.unwrap(newQ[:, c])
z['qpos'] = newQ; np.savez(out / 'reference.npz', **z)
ex = execute_reference(str(scene), str(out / 'reference.npz'), out / 'execution')
met = evaluate_files(man['source'], ex['trajectory'], epdir / 'input_manifest.json', ex['scene'], epdir / 'metric_geometry.json')
fmt = {k: (round(v['mean_position_error_cm'][0], 3), round(v['mean_orientation_error_degrees'][0], 2), round(v['max_position_error_m'][0] * 100, 1)) for k, v in met['official_style_reference_proxy']['objects'].items()}
res = dict(episode='episode_000025', input_dir=str(epdir), reference=str(out / 'reference.npz'), execution=ex, metrics=met, parent=str(src),
           seat_rotation=dict(T0=T0, T1=T1, DUR=DUR, gain=GAIN, tgain=TGAIN, lift=[LIFT, L0, L1, L2], translation_cm=(dtv * 100).tolist(), correction_rotvec_deg=(np.degrees(rv)).tolist(), full_correction_deg=float(np.degrees(Rc_full.magnitude()))), learned_policy=False)
(out / 'result.json').write_text(json.dumps(res, indent=1, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)) + '\n')
print(json.dumps(dict(out=str(out), mp_sr=met['reference_proxy_mp_sr'], objects=fmt, corr_deg=float(np.degrees(np.linalg.norm(rv))), trans_cm=(dtv * 100).round(2).tolist(), full_deg=float(np.degrees(Rc_full.magnitude())), verified=ex['physics_verified'])), flush=True)
