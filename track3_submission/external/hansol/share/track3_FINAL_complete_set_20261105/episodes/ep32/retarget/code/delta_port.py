"""Stream B: port a teacher reference (open-loop servo) from an old world into the target_final_v5 world by per-frame object deltas
(pattern streamR cup_delta_seed.py, generalized to both hands):
 - LEFT wrist: D_c(t) = minimal rotation of the old cup axis onto the new one about the old cup anchor + anchor translation (cup yaw ignored);
 - RIGHT wrist: full rigid spoon delta D_s(t) = T_new(t) T_old(t)^-1 (body frames), i.e. the wrist keeps its pose relative to the spoon target.
Both blended in from identity over [0, BLEND] s (frame 0 = scene initial state); --right-gain/--left-gain scale the delta (rotation by slerp,
translation linearly). Object columns = the new episode dir. Executes in SCENE (new world) and scores vs the new episode dir (proxy).
usage: delta_port.py SEED_REF.npz OUT_DIR SCENE OLD_EPD NEW_EPD [--blend 0.5] [--left-gain 1] [--right-gain 1] [--right-t0 0] [--no-exec]"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import sys, json, argparse
from pathlib import Path
import numpy as np, mujoco, trimesh
from scipy.spatial.transform import Rotation as R
LEG = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017'); sys.path[:0] = [str(LEG / 'code'), '/data1/SRL_V2D/third_party/DexAgent']
from tool_library.common import tracking as TR
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files
from wrist_reference import continuous_euler_xyz
ap = argparse.ArgumentParser(); ap.add_argument('seed'); ap.add_argument('out'); ap.add_argument('scene'); ap.add_argument('old'); ap.add_argument('new')
ap.add_argument('--blend', type=float, default=0.5); ap.add_argument('--left-gain', type=float, default=1.0); ap.add_argument('--right-gain', type=float, default=1.0)
ap.add_argument('--right-t0', type=float, default=0.0); ap.add_argument('--no-exec', action='store_true'); a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True); m = mujoco.MjModel.from_xml_path(a.scene)
ref = TR.load_reference(a.seed, m); Q = ref['qpos'].copy(); n = len(Q); fps = ref['fps']
qo = TR.load_reference(Path(a.old) / 'tracking_reference.npz', m)['qpos']; qn = TR.load_reference(Path(a.new) / 'tracking_reference.npz', m)['qpos']
oc = int(m.joint('blue_cup_free').qposadr[0]); sc = int(m.joint('wooden_spoon_free').qposadr[0])
src = json.loads((Path(a.new) / 'input_manifest.json').read_text())['source']
anc = trimesh.load(Path(src) / 'assets/blue_cup.obj', force='mesh', process=False).centroid.copy()
Q[:, oc:oc + 7] = qn[:, oc:oc + 7]; Q[:, sc:sc + 7] = qn[:, sc:sc + 7]
sm = lambda u: np.clip(u, 0, 1) ** 3 * (10 - 15 * np.clip(u, 0, 1) + 6 * np.clip(u, 0, 1) ** 2)
t = np.arange(n) / fps
wl = sm(t / a.blend) * a.left_gain; wr = sm((t - a.right_t0) / a.blend) * a.right_gain
adl = [int(m.joint(f'robot_left_left_{j}').qposadr[0]) for j in ('pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z')]
adr = [int(m.joint(f'robot_right_right_{j}').qposadr[0]) for j in ('pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z')]
for k in range(1, n):
    Ro, Rn = R.from_quat(qo[k, oc + 3:oc + 7], scalar_first=True), R.from_quat(qn[k, oc + 3:oc + 7], scalar_first=True)
    po, pn = qo[k, oc:oc + 3] + Ro.apply(anc), qn[k, oc:oc + 3] + Rn.apply(anc)
    ao, an = Ro.apply([0, 0, 1.]), Rn.apply([0, 0, 1.]); rv = np.cross(ao, an); s_ = np.linalg.norm(rv); ang = np.arctan2(s_, ao @ an)
    D = R.from_rotvec((rv / s_ * ang if s_ > 1e-9 else np.zeros(3)) * wl[k]); pw = Q[k, adl[:3]]; Rw = R.from_euler('XYZ', Q[k, adl[3:]])
    Q[k, adl[:3]] = po + D.apply(pw - po) + wl[k] * (pn - po); Q[k, adl[3:]] = continuous_euler_xyz(R.from_matrix((D * Rw).as_matrix()[None]), Q[k - 1, adl[3:]])[0]
    So, Sn = R.from_quat(qo[k, sc + 3:sc + 7], scalar_first=True), R.from_quat(qn[k, sc + 3:sc + 7], scalar_first=True)
    so, sn_ = qo[k, sc:sc + 3], qn[k, sc:sc + 3]; Ds = R.from_rotvec((Sn * So.inv()).as_rotvec() * wr[k])
    pw = Q[k, adr[:3]]; Rw = R.from_euler('XYZ', Q[k, adr[3:]])
    Q[k, adr[:3]] = so + Ds.apply(pw - so) + wr[k] * (sn_ - so); Q[k, adr[3:]] = continuous_euler_xyz(R.from_matrix((Ds * Rw).as_matrix()[None]), Q[k - 1, adr[3:]])[0]
refp = TR.save_reference(out / 'reference.npz', m, Q, ref['labels'], fps, ref['segment'], ref['segments'], source=f'delta_port of {a.seed} ({a.old} -> {a.new}); blend {a.blend}, gains L {a.left_gain} R {a.right_gain}, right_t0 {a.right_t0}', arms_coupled=False, floating_runtime=True)
if a.no_exec: print('wrote', refp); raise SystemExit
ex = execute_reference(a.scene, refp, out / 'execution'); mf = Path(a.new) / 'input_manifest.json'; mm = json.loads(mf.read_text())
mt = evaluate_files(mm['source'], ex['trajectory'], mf, ex['scene'], Path(a.new) / 'metric_geometry.json'); o = mt['official_style_reference_proxy']
res = dict(episode=mm['episode'], input_dir=str(Path(a.new).resolve()), reference=str(refp), execution=ex, metrics=mt, seed=a.seed, args=vars(a), script=__file__)
(out / 'result.json').write_text(json.dumps(res, indent=1, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)) + '\n')
print('RESULT', o['mp_sr'], {k: (round(v['mean_position_error_cm'][0], 2), round(v['mean_orientation_error_degrees'][0], 1)) for k, v in o['objects'].items()}, flush=True)
