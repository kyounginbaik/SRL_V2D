"""Stream R ep32 phase 6: position-only ILC of the LEFT wrist on an open-loop teacher (reference.npz) so the held/placed cup anchor follows the episode
dir's cup target: offset(t) <- offset(t) - GAIN * gaussian_smooth(achieved_anchor - target_anchor)(t) on frames in [T0, T1] (smooth 0.25 s edges), added
to the left pos_x/y/z commands; rotation and fingers untouched; the right hand untouched. Each iteration is a fresh physical execution; the best
iteration by max(cup cm/3, spoon cm/3, cup deg/30, spoon deg/30) against the episode dir is written as OUT/best (result.json + reference.npz).
usage: left_ilc32.py TEACHER_DIR OUT_DIR EPISODE_DIR [--gain 0.6] [--sigma 4] [--iters 4] [--t0 0.6] [--t1 12.9] [--zonly]"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import sys, json, argparse, shutil
from pathlib import Path
import numpy as np, mujoco, trimesh
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R
LEG = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017'); sys.path[:0] = [str(LEG / 'code'), '/data1/SRL_V2D/third_party/DexAgent']
from tool_library.common import tracking as TR
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files
ap = argparse.ArgumentParser(); ap.add_argument('teacher'); ap.add_argument('out'); ap.add_argument('epd'); ap.add_argument('--gain', type=float, default=0.6); ap.add_argument('--sigma', type=float, default=4)
ap.add_argument('--iters', type=int, default=4); ap.add_argument('--t0', type=float, default=0.6); ap.add_argument('--t1', type=float, default=12.9); ap.add_argument('--zonly', action='store_true'); a = ap.parse_args()
te, out, epd = Path(a.teacher), Path(a.out), Path(a.epd); out.mkdir(parents=True, exist_ok=True)
prev = json.loads((te / 'result.json').read_text()); scene = prev['execution']['scene']; m = mujoco.MjModel.from_xml_path(scene)
ref = TR.load_reference(te / 'reference.npz', m); Q0 = ref['qpos'].copy(); n = len(Q0); fps = ref['fps']; t = np.arange(n) / fps
tq = TR.load_reference(epd / 'tracking_reference.npz', m)['qpos']; oc = int(m.joint('blue_cup_free').qposadr[0])
anc = trimesh.load('/data1/SRL_V2D/challenge_reconstruction/final_recon/episode_000032/assets/blue_cup.obj', force='mesh', process=False).centroid.copy()
pr = np.array([tq[k, oc:oc + 3] + R.from_quat(tq[k, oc + 3:oc + 7], scalar_first=True).apply(anc) for k in range(n)])
adr = [int(m.joint(f'robot_left_left_pos_{j}').qposadr[0]) for j in 'xyz']
sm = lambda u: np.clip(u, 0, 1) ** 3 * (10 - 15 * np.clip(u, 0, 1) + 6 * np.clip(u, 0, 1) ** 2)
win = sm((t - a.t0) / 0.25) * (1 - sm((t - a.t1 + 0.25) / 0.25)) if a.t1 < t[-1] else sm((t - a.t0) / 0.25)
mf = epd / 'input_manifest.json'; mm = json.loads(mf.read_text())
def run(off, tag):
    Q = Q0.copy(); Q[:, adr] += off; Q[0] = Q0[0]; d = out / tag; d.mkdir(exist_ok=True)
    refp = TR.save_reference(d / 'reference.npz', m, Q, ref['labels'], fps, ref['segment'], ref['segments'], source=f'left-wrist position ILC {tag} on {te}', arms_coupled=False, floating_runtime=True)
    ex = execute_reference(scene, refp, d / 'execution'); mt = evaluate_files(mm['source'], ex['trajectory'], mf, ex['scene'], epd / 'metric_geometry.json')
    o = mt['official_style_reference_proxy']['objects']; ratio = max(max(v['mean_position_error_cm'][0] / 3, v['mean_orientation_error_degrees'][0] / 30) for v in o.values())
    (d / 'result.json').write_text(json.dumps(dict(episode='episode_000032', input_dir=str(epd.resolve()), reference=str(refp), execution=ex, metrics=mt, teacher=str(te), ilc=vars(a), iteration=tag), indent=1, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)) + '\n')
    tr = np.load(ex['trajectory']); ci = [str(x) for x in tr['object_names']].index('blue_cup')
    pa = np.array([tr['object_positions'][k, ci] + R.from_quat(tr['object_quaternions_wxyz'][k, ci][[1, 2, 3, 0]]).apply(anc) for k in range(n)])
    print(tag, 'ratio %.3f' % ratio, {k: (round(v['mean_position_error_cm'][0], 2), round(v['mean_orientation_error_degrees'][0], 1)) for k, v in o.items()}, flush=True); return ratio, pa - pr, d
off = np.zeros((n, 3)); best = None
for it in range(a.iters + 1):
    ratio, err, d = run(off, f'it{it}')
    if best is None or ratio < best[0]: best = (ratio, d)
    if it == a.iters: break
    g = gaussian_filter1d(err, a.sigma, axis=0, mode='nearest') * win[:, None]
    if a.zonly: g[:, :2] = 0
    off = off - a.gain * g
if (out / 'best').exists(): shutil.rmtree(out / 'best') if False else None
shutil.copytree(best[1], out / 'best', dirs_exist_ok=False) if not (out / 'best').exists() else None
print('BEST', best[1].name, round(best[0], 3))
