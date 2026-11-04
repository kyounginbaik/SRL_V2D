"""Stream R ep32: robot re-plan for the corrected spoon roll (pattern refinement_20261021_ep36/code/roll_replan.py). Rotates the RIGHT
wrist command of an open-loop teacher about the executed spoon long axis (through the executed spoon origin) by the per-frame
correction roll of a roll_correction.json (DIR=+1: as recorded; DIR=-1: the same final roll reached the other way round, phi-360*sign(phi));
fingers and the left hand unchanged. Executes in the teacher's unchanged scene; scores against the corrected and the original references.
--squeeze A: during the roll (weight = |phi|/max|phi|, dilated by 0.25 s) add A * (right finger hold posture at 5.0 s minus the pre-grasp
posture at 1.5 s) to the right finger commands (clipped to joint ranges) so the rolled pinch holds.
--robot-record JSON: take the ROBOT roll from another roll_correction.json (scoring still against CORRECTED_EPISODE_DIR);
--robot-sigma S: Gaussian-smooth the robot roll (S frames) so the wrist turns slower than the reference ramps.
usage: roll_replan32.py TEACHER_DIR OUT_DIR CORRECTED_EPISODE_DIR DIR [--gain G] [--squeeze A] [--robot-record JSON] [--robot-sigma S]"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import sys, json
from pathlib import Path
import numpy as np, mujoco
from scipy.spatial.transform import Rotation as R
LEG = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017'); sys.path[:0] = [str(LEG / 'code'), '/data1/SRL_V2D/third_party/DexAgent']
from tool_library.common import tracking as TR
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files
from wrist_reference import continuous_euler_xyz
a = sys.argv[1:]; gain = 1.0; squeeze = 0.0; rrec = None; rsig = 0.0
if '--robot-record' in a: i = a.index('--robot-record'); rrec = a[i + 1]; del a[i:i + 2]
if '--robot-sigma' in a: i = a.index('--robot-sigma'); rsig = float(a[i + 1]); del a[i:i + 2]
if '--squeeze' in a: i = a.index('--squeeze'); squeeze = float(a[i + 1]); del a[i:i + 2]
if '--gain' in a: i = a.index('--gain'); gain = float(a[i + 1]); del a[i:i + 2]
teacher, out, ced, sgn = Path(a[0]), Path(a[1]), Path(a[2]), float(a[3])
ORIG = LEG / 'refinement_20261018/expanded_batch/repaired_export_inputs/episode_000032'
man = json.loads((ced / 'input_manifest.json').read_text()); rec = json.loads(Path(man['spoon_roll_correction']['record']).read_text())
prev = json.loads((teacher / 'result.json').read_text()); scene = prev['execution']['scene']
m = mujoco.MjModel.from_xml_path(scene); ref = TR.load_reference(teacher / 'reference.npz', m); Q = ref['qpos'].copy(); fps = ref['fps']; n = len(Q)
phi = np.zeros(n)
for k, v in (json.loads(Path(rrec).read_text()) if rrec else rec)['per_frame_roll_deg'].items(): phi[int(k)] = v
if rsig:
    from scipy.ndimage import gaussian_filter1d
    phi = gaussian_filter1d(phi, rsig, mode='constant'); phi[np.abs(phi) < 0.5] = 0.
if sgn < 0: phi = np.where(phi != 0, phi - 360 * np.sign(phi) * (np.abs(phi) > 0), 0) * 1.0
phi *= gain
ex = np.load(teacher / 'execution/trajectory.npz'); on = [str(x) for x in ex['object_names']]; si = on.index('wooden_spoon')
ps = ex['object_positions'][:n, si]; Rs = R.from_quat(ex['object_quaternions_wxyz'][:n, si][:, [1, 2, 3, 0]])
adr = [int(m.joint(f'robot_right_right_{j}').qposadr[0]) for j in ('pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z')]
for k in range(n):
    if phi[k] == 0: continue
    Rc = Rs[k] * R.from_euler('x', phi[k], degrees=True) * Rs[k].inv(); c = ps[k]
    p = Q[k, adr[:3]]; Rw = R.from_euler('XYZ', Q[k, adr[3:]])
    Q[k, adr[:3]] = c + Rc.apply(p - c); Rn = Rc * Rw
    Q[k, adr[3:]] = continuous_euler_xyz(R.from_matrix(Rn.as_matrix()[None]), Q[k - 1, adr[3:]])[0]
if squeeze:
    fj = [j for j in range(m.njnt) if (m.joint(j).name or '').startswith('robot_right_right_') and any(f'_{x}_' in m.joint(j).name for x in ('thumb', 'index', 'middle', 'ring', 'pinky'))]
    fa = m.jnt_qposadr[fj]; rng = m.jnt_range[fj]; close = Q[100, fa] - Q[30, fa]
    w = np.abs(phi) / max(np.abs(phi).max(), 1e-9); w = np.array([w[max(0, k - 5):k + 6].max() for k in range(n)])
    Q[:, fa] = np.clip(Q[:, fa] + squeeze * w[:, None] * close[None], rng[:, 0], rng[:, 1])
out.mkdir(parents=True, exist_ok=True)
refp = TR.save_reference(out / 'reference.npz', m, Q, ref['labels'], fps, ref['segment'], ref['segments'], source=f'right wrist rotated about the held spoon long axis by the corrected roll (dir {sgn:+.0f}, gain {gain}, squeeze {squeeze}); fingers and other commands unchanged', arms_coupled=False, floating_runtime=True)
execution = execute_reference(scene, refp, out / 'execution')
res = dict(episode='episode_000032', input_dir=str(ced), reference=str(refp), execution=execution, teacher=str(teacher), roll=dict(direction=sgn, gain=gain, squeeze=squeeze, robot_record=rrec, robot_sigma=rsig, record=man['spoon_roll_correction']['record']))
for tag, ed in (('corrected', ced), ('original', ORIG)):
    mm = json.loads((ed / 'input_manifest.json').read_text())
    mt = evaluate_files(mm['source'], execution['trajectory'], ed / 'input_manifest.json', execution['scene'], ed / 'metric_geometry.json'); res['metrics' if tag == 'corrected' else 'metrics_original'] = mt
    o = mt['official_style_reference_proxy']['objects']
    print(tag, 'MP-SR', mt['reference_proxy_mp_sr'], {k: (round(v['mean_position_error_cm'][0], 2), round(v['mean_orientation_error_degrees'][0], 1)) for k, v in o.items()}, flush=True)
(out / 'result.json').write_text(json.dumps(res, indent=1, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)) + '\n')
