"""Stream Q phase 6: object-relative ILC step for the insertion. From an executed source rollout: in-hand transform H of board 2 in
the right wrist (mean over [T0-0.5, T0]); desired board-2 pose D(t) = B1_sim(t) inv(B1_ref(t)) B2_ref(t) (the reference's exact
interlock relative pose applied to where board 1 actually is); desired wrist W*(t) = D(t) H^-1; command update
Wcmd <- (W*(t) Wact(t)^-1)^(g w(t)) Wcmd (6-DoF, w: smoothstep T0->T1, 1 after). Fingers unchanged. Executes + scores.
usage: seat_follow.py SRC_DIR WORLD_DIR OUT_DIR T0 T1 GAIN [SCENE]"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import sys, json
from pathlib import Path
import numpy as np, mujoco
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import gaussian_filter1d
LEG = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017')
sys.path[:0] = [str(LEG / 'code'), '/data1/SRL_V2D/third_party/DexAgent']
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files
src, world, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve(); T0, T1, G = map(float, sys.argv[4:7])
scene = Path(sys.argv[7]) if len(sys.argv) > 7 else world / 'scene/scene_physics_C.xml'; epdir = world / 'episodes/episode_000025'
man = json.loads((epdir / 'input_manifest.json').read_text()); out.mkdir(parents=True, exist_ok=False)
z = {k: v for k, v in np.load(src / 'reference.npz', allow_pickle=False).items()}; Q = np.asarray(z['qpos'], float).copy(); names = list(np.asarray(z['joint_names']).astype(str))
tr = np.load(json.loads((src / 'result.json').read_text())['execution']['trajectory']); qa = tr['qpos']
Tref = np.load(world / 'source_corrected/episode_000025/object_poses.npz')['world_from_object']
m = mujoco.MjModel.from_xml_path(str(scene)); d = mujoco.MjData(m); bid = m.jnt_bodyid[m.joint('robot_right_right_rot_z').id]
cols = [names.index(f'robot_right_right_{a}') for a in ('pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z')]
def H4(Rm, p): M = np.eye(4); M[:3, :3] = Rm; M[:3, 3] = p; return M
def wrist(q): d.qpos[:] = q; mujoco.mj_kinematics(m, d); return H4(d.xmat[bid].reshape(3, 3).copy(), d.xpos[bid].copy())
def obj(k, j): return H4(R.from_quat(tr['object_quaternions_wxyz'][k, j][[1, 2, 3, 0]]).as_matrix(), tr['object_positions'][k, j])
n = len(Q); kk0, kk1 = int((T0 - .5) * 20), int(T0 * 20)
Hs = [np.linalg.inv(wrist(qa[k])) @ obj(k, 1) for k in range(kk0, kk1 + 1)]
Hr = R.from_matrix([h[:3, :3] for h in Hs]).mean().as_matrix(); Hp = np.mean([h[:3, 3] for h in Hs], 0); H = H4(Hr, Hp)
rv = np.zeros((n, 3)); dp = np.zeros((n, 3)); tt = np.arange(n) / 20.
for k in range(n):
    if tt[k] < T0: continue
    D = obj(k, 0) @ np.linalg.inv(Tref[k, 0]) @ Tref[k, 1]; Ws = D @ np.linalg.inv(H); Wa = wrist(qa[k])
    rv[k] = R.from_matrix(Ws[:3, :3] @ Wa[:3, :3].T).as_rotvec(); dp[k] = Ws[:3, 3] - Wa[:3, 3]
rv = gaussian_filter1d(rv, 2.0, axis=0); dp = gaussian_filter1d(dp, 2.0, axis=0)
sm = lambda u: float(np.clip(u, 0, 1)) ** 2 * (3 - 2 * float(np.clip(u, 0, 1)))
for k in range(n):
    if tt[k] < T0: continue
    w = G * (1. if tt[k] >= T1 else sm((tt[k] - T0) / (T1 - T0)))
    Rk = R.from_rotvec(w * rv[k]); Wc = R.from_euler('XYZ', Q[k, cols[3:6]])
    # rotate the command about the actual wrist origin, then translate
    Q[k, cols[3:6]] = (Rk * Wc).as_euler('XYZ'); Q[k, cols[:3]] = Q[k, cols[:3]] + w * dp[k]
for c in cols[3:6]: Q[:, c] = np.unwrap(Q[:, c])
z['qpos'] = Q; np.savez(out / 'reference.npz', **z)
ex = execute_reference(str(scene), str(out / 'reference.npz'), out / 'execution')
met = evaluate_files(man['source'], ex['trajectory'], epdir / 'input_manifest.json', ex['scene'], epdir / 'metric_geometry.json')
fmt = {k: (round(v['mean_position_error_cm'][0], 3), round(v['mean_orientation_error_degrees'][0], 2), round(v['max_position_error_m'][0] * 100, 1)) for k, v in met['official_style_reference_proxy']['objects'].items()}
res = dict(episode='episode_000025', input_dir=str(epdir), reference=str(out / 'reference.npz'), execution=ex, metrics=met, parent=str(src), seat_follow=dict(T0=T0, T1=T1, gain=G, max_rot_deg=float(np.degrees(np.linalg.norm(rv, axis=1).max())), max_dp_cm=float(np.linalg.norm(dp, axis=1).max() * 100)), learned_policy=False)
(out / 'result.json').write_text(json.dumps(res, indent=1, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)) + '\n')
tr2 = np.load(ex['trajectory']); e2 = [np.linalg.norm(tr2['object_positions'][k, 1] - Tref[k, 1, :3, 3]) * 100 for k in range(300, n, 10)]
print(json.dumps(dict(out=str(out), mp_sr=met['reference_proxy_mp_sr'], objects=fmt, b2_err_cm_15to17s=np.round(e2, 1).tolist(), **res['seat_follow'])), flush=True)
