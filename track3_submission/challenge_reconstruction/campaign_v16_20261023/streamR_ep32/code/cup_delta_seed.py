"""Stream R ep32 phase 6: move a teacher's LEFT wrist commands with the cup: per frame D(t) = (minimal rotation carrying OUR reference cup axis onto the
COMPOSE cup axis, about our cup anchor) followed by the anchor translation to the compose anchor; wrist_new = D(t) o wrist(t). The symmetric cup yaw is
ignored. D is blended in from identity over [0, BLEND] s (frame 0 = scene initial state). Optionally replace the RIGHT hand by another reference's right
hand (e.g. a pickup-CEM teacher) for frames >= 1. Executes the result in SCENE and scores vs the corrected episode dir.
usage: cup_delta_seed.py SEED_REF.npz OUT_DIR SCENE ORIG_EPD CORR_EPD [--blend 0.3] [--right-from REF.npz] [--gain 1.0]"""
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
ap = argparse.ArgumentParser(); ap.add_argument('seed'); ap.add_argument('out'); ap.add_argument('scene'); ap.add_argument('orig'); ap.add_argument('corr')
ap.add_argument('--blend', type=float, default=0.3); ap.add_argument('--right-from'); ap.add_argument('--gain', type=float, default=1.0); a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True); m = mujoco.MjModel.from_xml_path(a.scene)
ref = TR.load_reference(a.seed, m); Q = ref['qpos'].copy(); n = len(Q); fps = ref['fps']
qo = TR.load_reference(Path(a.orig) / 'tracking_reference.npz', m)['qpos']; qc = TR.load_reference(Path(a.corr) / 'tracking_reference.npz', m)['qpos']
oc = int(m.joint('blue_cup_free').qposadr[0]); anc = trimesh.load('/data1/SRL_V2D/challenge_reconstruction/final_recon/episode_000032/assets/blue_cup.obj', force='mesh', process=False).centroid.copy()
Q[:, oc:oc + 7] = qc[:, oc:oc + 7]; sc = int(m.joint('wooden_spoon_free').qposadr[0]); Q[:, sc:sc + 7] = qc[:, sc:sc + 7]
if a.right_from:
    rr = TR.load_reference(a.right_from, m)['qpos']; cols = [int(m.jnt_qposadr[j]) for j in range(m.njnt) if (m.joint(j).name or '').startswith('robot_right_')]; Q[1:, cols] = rr[1:, cols]
adr = [int(m.joint(f'robot_left_left_{j}').qposadr[0]) for j in ('pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z')]
t = np.arange(n) / fps; u = np.clip(t / a.blend, 0, 1); w = (u ** 3 * (10 - 15 * u + 6 * u * u)) * a.gain
for k in range(1, n):
    Ro, Rc = R.from_quat(qo[k, oc + 3:oc + 7], scalar_first=True), R.from_quat(qc[k, oc + 3:oc + 7], scalar_first=True)
    po, pc = qo[k, oc:oc + 3] + Ro.apply(anc), qc[k, oc:oc + 3] + Rc.apply(anc)
    ao, ac = Ro.apply([0, 0, 1.]), Rc.apply([0, 0, 1.]); rv = np.cross(ao, ac); s_ = np.linalg.norm(rv); ang = np.arctan2(s_, ao @ ac)
    D = R.from_rotvec((rv / s_ * ang if s_ > 1e-9 else np.zeros(3)) * w[k]); pw = Q[k, adr[:3]]; Rw = R.from_euler('XYZ', Q[k, adr[3:]])
    Q[k, adr[:3]] = po + D.apply(pw - po) + w[k] * (pc - po); Q[k, adr[3:]] = continuous_euler_xyz(R.from_matrix((D * Rw).as_matrix()[None]), Q[k - 1, adr[3:]])[0]
refp = TR.save_reference(out / 'reference.npz', m, Q, ref['labels'], fps, ref['segment'], ref['segments'], source=f'cup-delta left wrist (blend {a.blend}s, gain {a.gain}); right from {a.right_from}', arms_coupled=False, floating_runtime=True)
ex = execute_reference(a.scene, refp, out / 'execution'); mf = Path(a.corr) / 'input_manifest.json'; mm = json.loads(mf.read_text())
mt = evaluate_files(mm['source'], ex['trajectory'], mf, ex['scene'], Path(a.corr) / 'metric_geometry.json'); o = mt['official_style_reference_proxy']
res = dict(episode='episode_000032', input_dir=str(Path(a.corr).resolve()), reference=str(refp), execution=ex, metrics=mt, seed=a.seed, right_from=a.right_from, blend=a.blend, gain=a.gain)
(out / 'result.json').write_text(json.dumps(res, indent=1, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)) + '\n')
print('RESULT', o['mp_sr'], {k: (round(v['mean_position_error_cm'][0], 2), round(v['mean_orientation_error_degrees'][0], 1)) for k, v in o['objects'].items()}, flush=True)
