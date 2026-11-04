"""per-second object error of an execution trajectory vs the world tracking reference. usage: timeline.py RUN_DIR EPDIR"""
import sys, numpy as np, mujoco
from pathlib import Path
from scipy.spatial.transform import Rotation as R
sys.path[:0] = ['/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code', '/data1/SRL_V2D/third_party/DexAgent']
from tool_library.common import tracking as TR
run, ep = Path(sys.argv[1]), Path(sys.argv[2])
t = np.load(run / 'execution/trajectory.npz'); m = mujoco.MjModel.from_xml_path(str(run / 'execution/scene_physics.xml'))
ref = TR.load_reference(ep / 'tracking_reference.npz', m)['qpos']; q = t['qpos'] if 'qpos' in t.files else None
if q is None: print(t.files); sys.exit()
fps = 20; n = len(ref); step = max(1, round((len(q) - 1) / (n - 1))) if len(q) > n else 1
q = q[::step][:n] if len(q) > n else q
for j in range(m.njnt):
    if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE: continue
    a = int(m.jnt_qposadr[j]); nm = m.joint(j).name
    p = np.linalg.norm(q[:, a:a+3] - ref[:len(q), a:a+3], axis=1) * 100
    d = np.degrees((R.from_quat(q[:, a+3:a+7], scalar_first=True) * R.from_quat(ref[:len(q), a+3:a+7], scalar_first=True).inv()).magnitude())
    print(nm, 'mean %.2f/%.1f' % (p.mean(), d.mean()), 'per-s cm', np.round([p[s*fps:(s+1)*fps].mean() for s in range(int(np.ceil(len(q)/fps)))], 1))
    print(' ' * len(nm), '           per-s deg', np.round([d[s*fps:(s+1)*fps].mean() for s in range(int(np.ceil(len(q)/fps)))], 0))
