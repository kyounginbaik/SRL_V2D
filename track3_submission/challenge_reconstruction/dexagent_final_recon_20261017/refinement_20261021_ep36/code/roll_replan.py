"""Episode 36 robot re-plan for the corrected spoon roll: rotate the RIGHT wrist command about the spoon's executed long axis
(through the executed spoon origin) by phi(t) (0->SIGN*180 deg over [A0,A1], back over [B0,B1]) so the held spoon follows the
corrected roll; fingers unchanged. Executes in the teacher's unchanged scene and scores against the corrected and original
references. usage: roll_replan.py TEACHER_DIR OUT_DIR SIGN [A0 A1 B0 B1] [--source-traj TRAJ.npz]"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import sys, json
from pathlib import Path
import numpy as np, mujoco
from scipy.spatial.transform import Rotation as R
LEG = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017'); E = LEG / 'refinement_20261021_ep36'
sys.path[:0] = [str(LEG / 'code'), '/data1/SRL_V2D/third_party/DexAgent']
from tool_library.common import tracking as TR
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files
from wrist_reference import continuous_euler_xyz
a = sys.argv[1:]; srct = None
if '--source-traj' in a: i = a.index('--source-traj'); srct = a[i + 1]; del a[i:i + 2]
teacher, out, sign = Path(a[0]), Path(a[1]), float(a[2]); A0, A1, B0, B1 = (map(float, a[3:7]) if len(a) >= 7 else (12.35, 12.60, 14.65, 14.90))
prev = json.loads((teacher / 'result.json').read_text()); scene = prev['execution']['scene']
m = mujoco.MjModel.from_xml_path(scene); ref = TR.load_reference(teacher / 'reference.npz', m); Q = ref['qpos'].copy(); fps = ref['fps']; n = len(Q); t = np.arange(n) / fps
ex = np.load(srct or (teacher / 'execution/trajectory.npz')); ps = ex['object_positions'][:n, 0]; Rs = R.from_quat(ex['object_quaternions_wxyz'][:n, 0][:, [1, 2, 3, 0]])
s = lambda u: np.clip(u, 0, 1) ** 3 * (10 - 15 * np.clip(u, 0, 1) + 6 * np.clip(u, 0, 1) ** 2)
phi = sign * 180.0 * (s((t - A0) / (A1 - A0)) - s((t - B0) / (B1 - B0)))
adr = [int(m.joint(f'robot_right_right_{j}').qposadr[0]) for j in ('pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z')]
for k in range(n):
    if phi[k] == 0: continue
    Rc = Rs[k] * R.from_euler('x', phi[k], degrees=True) * Rs[k].inv(); c = ps[k]
    p = Q[k, adr[:3]]; Rw = R.from_euler('XYZ', Q[k, adr[3:]])
    Q[k, adr[:3]] = c + Rc.apply(p - c); Rn = Rc * Rw
    Q[k, adr[3:]] = continuous_euler_xyz(R.from_matrix(Rn.as_matrix()[None]), Q[k - 1, adr[3:]])[0]
out.mkdir(parents=True, exist_ok=True)
refp = TR.save_reference(out / 'reference.npz', m, Q, ref['labels'], fps, ref['segment'], ref['segments'], source=f'right wrist rotated about the held spoon long axis by the corrected roll ({sign:+.0f}*180 deg ramps); fingers and other commands unchanged', arms_coupled=False, floating_runtime=True)
execution = execute_reference(scene, refp, out / 'execution')
res = dict(episode='episode_000036', input_dir=str(E / 'episode_corrected/episode_000036'), reference=str(refp), execution=execution, teacher=str(teacher), roll=dict(sign=sign, ramps=[[A0, A1], [B0, B1]]))
for tag, src, ed in (('corrected', E / 'source_corrected/episode_000036', E / 'episode_corrected/episode_000036'), ('original', Path('/data1/SRL_V2D/challenge_reconstruction/final_recon/episode_000036'), LEG / 'episodes/episode_000036')):
    mt = evaluate_files(src, execution['trajectory'], ed / 'input_manifest.json', execution['scene'], ed / 'metric_geometry.json'); res['metrics' if tag == 'corrected' else 'metrics_original'] = mt
    o = mt['official_style_reference_proxy']['objects']['wooden_spoon']; print(tag, 'MP-SR', mt['reference_proxy_mp_sr'], round(o['mean_position_error_cm'][0], 2), 'cm', round(o['mean_orientation_error_degrees'][0], 1), 'deg', flush=True)
(out / 'result.json').write_text(json.dumps(res, indent=1, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)) + '\n')
