"""Stream P ep33 phase 6: teacher-level ILC for a compose-world grasp teacher (open-loop servo reference, physics_pickup_search_v2
output dir with reference.npz + result.json). Each iteration: from the executed trajectory, the smoothed spoon error vs the compose
reference (body-frame rotation rotvec ref->achieved and world position) on [t0, t1] is corrected by moving the RIGHT wrist reference
about the executed spoon origin: rotation by -g_rot * err (roll = body x only unless --all-axes), translation by -g_pos * err_pos;
then the unchanged floating_runtime.execute_reference re-executes and evaluate_files scores against the compose episode dir.
usage: teacher_ilc33.py TEACHER_DIR OUT_DIR --t0 7.5 --t1 14.3 --g-rot 0.8 --g-pos 0.5 [--iterations 2] [--sigma 0.4] [--all-axes]"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import sys, json, argparse, shutil, numpy as np, mujoco
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R
LEG = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017'); sys.path[:0] = [str(LEG / 'code'), '/data1/SRL_V2D/third_party/DexAgent']
from tool_library.common import tracking as TR
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files
from wrist_reference import continuous_euler_xyz
P = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamP_ep33'); OEP = P / 'staging_C/episode_compose/episode_000033'
sm = lambda u: np.clip(u, 0, 1) ** 3 * (10 - 15 * np.clip(u, 0, 1) + 6 * np.clip(u, 0, 1) ** 2)
ap = argparse.ArgumentParser(); ap.add_argument('teacher'); ap.add_argument('out'); ap.add_argument('--t0', type=float, required=True); ap.add_argument('--t1', type=float, required=True)
ap.add_argument('--g-rot', type=float, default=0.8); ap.add_argument('--g-pos', type=float, default=0.5); ap.add_argument('--iterations', type=int, default=2); ap.add_argument('--sigma', type=float, default=0.4); ap.add_argument('--all-axes', action='store_true')
ap.add_argument('--l-squeeze', default='', help='left finger flexion closing bias amt_rad,t0,t1 added to the reference (0.25 s ramps), applied in every iteration');
ap.add_argument('--r-freeze', default='', help='t0,t1: no right-hand correction in this window (smoothstep 0.3 s)');
ap.add_argument('--r-offset', default='', help='dx,dy,dz_cm,t0,t1: windowed right-wrist world translation added in iteration 0 (0.3 s ramps)');
ap.add_argument('--r-cupout', default='', help='dist_cm,t0,t1: windowed right-wrist translation along the executed cup opening axis (out of the cup), iteration 0');
ap.add_argument('--l-pos', type=float, default=0.0); ap.add_argument('--l-rot', type=float, default=0.0); ap.add_argument('--l-t0', type=float, default=2.6); ap.add_argument('--l-t1', type=float, default=16.0)
a = ap.parse_args(); src = Path(a.teacher); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
man = json.loads((OEP / 'input_manifest.json').read_text()); Wall = np.asarray(man['setup_from_source']) @ np.load(Path(man['source']) / 'object_poses.npz')['world_from_object']; Wt = Wall[:, 1]; Wc = Wall[:, 0]
CUPAX = np.array([-0.0237963577378219, -0.005046973670649213, 0.9997040869253169])
prev = src; hist = []
for it in range(a.iterations):
    r = json.loads((prev / 'result.json').read_text()); scene = r['execution']['scene']; m = mujoco.MjModel.from_xml_path(scene)
    ref = TR.load_reference(prev / 'reference.npz', m); Q = ref['qpos'].copy(); n = len(Q); ft = np.arange(n) / ref['fps']; tr = np.load(r['execution']['trajectory'])
    si = [str(x) for x in tr['object_names']].index('wooden_spoon'); ps = tr['object_positions'][:n, si]; Rs = R.from_quat(tr['object_quaternions_wxyz'][:n, si][:, [1, 2, 3, 0]])
    erv = (R.from_matrix(Wt[:n, :3, :3]).inv() * Rs).as_rotvec()            # body-frame error ref->achieved
    if not a.all_axes: erv[:, 1:] = 0
    erv = gaussian_filter1d(erv, a.sigma * 20, axis=0, mode='nearest'); ep = gaussian_filter1d(ps - Wt[:n, :3, 3], a.sigma * 20, axis=0, mode='nearest')
    g = sm((ft - a.t0) / 0.3) * (1 - sm((ft - (a.t1 - 0.3)) / 0.3))
    if a.r_freeze:
        z0, z1 = [float(x) for x in a.r_freeze.split(',')]; g = g * (1 - sm((ft - z0 + 0.3) / 0.3) * (1 - sm((ft - z1) / 0.3)))
    if a.r_offset and it == 0:
        ox_, oy_, oz_, w0, w1 = [float(x) for x in a.r_offset.split(',')]; go = sm((ft - w0) / 0.3) * (1 - sm((ft - (w1 - 0.3)) / 0.3))
        adr0 = [int(m.joint(f'robot_right_right_{j}').qposadr[0]) for j in ('pos_x', 'pos_y', 'pos_z')]; Q[:, adr0] += go[:, None] * np.array([ox_, oy_, oz_]) / 100
    if a.r_cupout and it == 0:
        dist, w0, w1 = [float(x) for x in a.r_cupout.split(',')]; go = sm((ft - w0) / 0.3) * (1 - sm((ft - (w1 - 0.3)) / 0.3))
        ci_ = [str(x) for x in tr['object_names']].index('blue_cup'); Rc_ = R.from_quat(tr['object_quaternions_wxyz'][:n, ci_][:, [1, 2, 3, 0]]).as_matrix()
        u = gaussian_filter1d(-Rc_[:, :, 2], 4, axis=0); u /= np.linalg.norm(u, axis=1, keepdims=True)
        adr0 = [int(m.joint(f'robot_right_right_{j}').qposadr[0]) for j in ('pos_x', 'pos_y', 'pos_z')]; Q[:, adr0] += (go * dist / 100)[:, None] * u
    if a.l_squeeze and it == 0:
        from tool_library.common import hand as Hh
        amt, q0, q1 = [float(x) for x in a.l_squeeze.split(',')]; rigL = Hh.HandRig.build(m, 'robot_left_', 'left'); gq = sm((ft - q0) / 0.25) * (1 - sm((ft - (q1 - 0.25)) / 0.25))
        for fi in ('thumb', 'index', 'middle', 'ring', 'pinky'):
            for jn in rigL.flexion_joints(fi):
                col = rigL.finger_qpos[jn]; lo, hi = rigL.finger_range[jn]; Q[:, col] = np.clip(Q[:, col] + amt * gq, lo, hi)
    adr = [int(m.joint(f'robot_right_right_{j}').qposadr[0]) for j in ('pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z')]
    for k in range(n):
        if g[k] == 0: continue
        C = Rs[k] * R.from_rotvec(-a.g_rot * g[k] * erv[k]) * Rs[k].inv(); c = ps[k]
        Q[k, adr[:3]] = c + C.apply(Q[k, adr[:3]] - c) - a.g_pos * g[k] * ep[k]
        Q[k, adr[3:]] = continuous_euler_xyz(R.from_matrix((C * R.from_euler('XYZ', Q[k, adr[3:]])).as_matrix()[None]), Q[k, adr[3:]])[0]
    if a.l_pos or a.l_rot:   # left hand: cup position error and axis-tilt error (cup axially symmetric)
        ci = [str(x) for x in tr['object_names']].index('blue_cup'); pc = tr['object_positions'][:n, ci]; Rc = R.from_quat(tr['object_quaternions_wxyz'][:n, ci][:, [1, 2, 3, 0]])
        u = Rc.apply(CUPAX); v = np.einsum('nij,j->ni', Wc[:n, :3, :3], CUPAX); cax = np.cross(v, u); sn = np.linalg.norm(cax, axis=1, keepdims=True)
        ang = np.arctan2(sn[:, 0], (u * v).sum(1)); erc = np.where(sn > 1e-9, cax / np.maximum(sn, 1e-9), 0) * ang[:, None]   # world rotvec ref->achieved axis
        erc = gaussian_filter1d(erc, a.sigma * 20, axis=0, mode='nearest'); epc = gaussian_filter1d(pc - Wc[:n, :3, 3], a.sigma * 20, axis=0, mode='nearest')
        gl = sm((ft - a.l_t0) / 0.3) * (1 - sm((ft - (a.l_t1 - 0.3)) / 0.3)); adl = [int(m.joint(f'robot_left_left_{j}').qposadr[0]) for j in ('pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z')]
        for k in range(n):
            if gl[k] == 0: continue
            C = R.from_rotvec(-a.l_rot * gl[k] * erc[k]); c = pc[k]
            Q[k, adl[:3]] = c + C.apply(Q[k, adl[:3]] - c) - a.l_pos * gl[k] * epc[k]
            Q[k, adl[3:]] = continuous_euler_xyz(R.from_matrix((C * R.from_euler('XYZ', Q[k, adl[3:]])).as_matrix()[None]), Q[k, adl[3:]])[0]
    d = out / f'iter{it}'; d.mkdir(exist_ok=True)
    refp = TR.save_reference(d / 'reference.npz', m, Q, ref['labels'], ref['fps'], ref['segment'], ref['segments'], source=f'teacher_ilc33 iter {it} from {prev}', arms_coupled=False, floating_runtime=True)
    ex = execute_reference(scene, refp, d / 'execution')
    met = evaluate_files(man['source'], ex['trajectory'], OEP / 'input_manifest.json', ex['scene'], OEP / 'metric_geometry.json')
    o = {k: (round(v['mean_position_error_cm'][0], 3), round(v['mean_orientation_error_degrees'][0], 2)) for k, v in met['official_style_reference_proxy']['objects'].items()}
    res = dict(r); res.update(execution=ex, metrics=met, teacher_ilc=dict(previous=str(prev), t0=a.t0, t1=a.t1, g_rot=a.g_rot, g_pos=a.g_pos, sigma=a.sigma, all_axes=a.all_axes, l_pos=a.l_pos, l_rot=a.l_rot, l_squeeze=a.l_squeeze, r_freeze=a.r_freeze, r_offset=a.r_offset, r_cupout=a.r_cupout, l_window=[a.l_t0, a.l_t1], script=__file__))
    (d / 'result.json').write_text(json.dumps(res, indent=1, default=str)); hist.append(dict(iteration=it, objects=o, mp_sr=met['official_style_reference_proxy']['mp_sr'])); print('ITER', it, json.dumps(hist[-1]), flush=True)
    prev = d
(out / 'history.json').write_text(json.dumps(hist, indent=1))
