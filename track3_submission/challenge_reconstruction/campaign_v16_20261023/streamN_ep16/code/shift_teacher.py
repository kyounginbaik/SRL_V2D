"""Transplant an old-world teacher into the re-targeted world: the right wrist translation commands are shifted by the constant
world offset d0 = p_new_ref(0) - p_old_ref(0) of the spoon at rest (the spoon starts d0 away in the new reference, so the pinch lands at
the same spoon-relative spot), optionally blended out after T_OUT s (ramp 0.5 s). Robot-only change; objects/clock untouched.
usage: shift_teacher.py TEACHER_DIR OLD_WORLD NEW_WORLD OUT_DIR [T_OUT|none]"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import sys, json
from pathlib import Path
import numpy as np, mujoco
sys.path[:0]=['/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code','/data1/SRL_V2D/third_party/DexAgent']
from tool_library.common import tracking as TR, hand as H
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files
te, ow, nw, out = (Path(x) for x in sys.argv[1:5]); tout = None if len(sys.argv) < 6 or sys.argv[5] == 'none' else float(sys.argv[5])
scene = str(nw / 'scene/controller_scene.xml'); m = mujoco.MjModel.from_xml_path(scene); epd = nw / 'episodes/episode_000016'
r = json.loads((te / 'result.json').read_text()); ref = TR.load_reference(te / 'reference.npz', m); Q = ref['qpos'].copy(); fps = ref['fps']; T = len(Q); t = np.arange(T) / fps
oc = int(m.joint('wooden_spoon_free').qposadr[0]); rc = int(m.joint('plastic_dish_rack_free').qposadr[0])
on = TR.load_reference(epd / 'tracking_reference.npz', m)['qpos']; oo = TR.load_reference(ow / 'episodes/episode_000016/tracking_reference.npz', m)['qpos']
d0 = on[0, oc:oc + 3] - oo[0, oc:oc + 3]
rig = H.HandRig.build(m, 'robot_right_', 'right'); wc = np.array([rig.wrist_qpos[k] for k in ('x', 'y', 'z')])
sm = lambda u: np.clip(u, 0, 1) ** 3 * (10 - 15 * np.clip(u, 0, 1) + 6 * np.clip(u, 0, 1) ** 2)
g = np.ones(T) if tout is None else 1 - sm((t - tout) / 0.5)
Q[:, wc] += g[:, None] * d0; Q[:, oc:oc + 7] = on[:, oc:oc + 7]; Q[:, rc:rc + 7] = on[:, rc:rc + 7]; Q[0, wc] = Q[0, wc]
out.mkdir(parents=True, exist_ok=False)
p = TR.save_reference(out / 'reference.npz', m, Q, ref['labels'], fps, ref['segment'], ref['segments'], source=f'shift_teacher d0={d0.tolist()} t_out={tout} on {te}', floating_runtime=True, no_retiming=True)
ex = execute_reference(scene, p, out / 'execution'); man = json.loads((epd / 'input_manifest.json').read_text())
met = evaluate_files(man['source'], ex['trajectory'], epd / 'input_manifest.json', ex['scene'], epd / 'metric_geometry.json')
(out / 'result.json').write_text(json.dumps(dict(episode='episode_000016', input_dir=str(epd), reference=str(p), execution=ex, metrics=met, source=str(te), shift=dict(d0_m=d0.tolist(), t_out=tout), learned_policy=False), indent=1, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)) + '\n')
print(json.dumps(dict(out=str(out), d0_cm=(d0 * 100).round(2).tolist(), mp_sr=met['reference_proxy_mp_sr'], objects={k: (round(v['mean_position_error_cm'][0], 2), round(v['mean_orientation_error_degrees'][0], 1), round(v['max_position_error_m'][0] * 100, 1)) for k, v in met['official_style_reference_proxy']['objects'].items()})))
