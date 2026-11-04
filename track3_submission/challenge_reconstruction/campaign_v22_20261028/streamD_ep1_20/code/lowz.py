"""lowest mesh z (cm) per second for the physics reference and the G0-mapped target. usage: lowz.py EP EPISODE_DIR ANCHOR ASSET_DIR [EXEC_TRAJ: anchor on its achieved frame 0 instead of the tracking reference; also prints achieved]"""
import sys, numpy as np, trimesh
from pathlib import Path
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, str(Path(__file__).resolve().parent)); import dlib as L
ep = int(sys.argv[1]); epd = Path(sys.argv[2]); A = sys.argv[3]; AD = Path(sys.argv[4]); names = L.NAMES[ep]
t = np.load(epd / 'tracking_reference.npz', allow_pickle=False); jc = [str(x) for x in t['joint_names']]
def col(o): return jc.index(f'{o}_free[0]') if f'{o}_free[0]' in jc else jc.index(f'{o}_free')
old = {}
for o in names:
    c = col(o); q = t['qpos']; T = np.tile(np.eye(4), (len(q), 1, 1)); T[:, :3, 3] = q[:, c:c + 3]; T[:, :3, :3] = R.from_quat(q[:, c + 3:c + 7], scalar_first=True).as_matrix(); old[o] = T
ach = {}
if len(sys.argv) > 5:
    tr = np.load(sys.argv[5]); nm = [str(x) for x in tr['object_names']]
    for o in names:
        i = nm.index(o); T = np.tile(np.eye(4), (len(tr['object_positions']), 1, 1)); T[:, :3, 3] = tr['object_positions'][:, i]; T[:, :3, :3] = R.from_quat(tr['object_quaternions_wxyz'][:, i][:, [1, 2, 3, 0]]).as_matrix(); ach[o] = T
Tc, X = L.target_ourbody(ep); G0 = (ach or old)[A][0] @ np.linalg.inv(Tc[A][0])
for o in names:
    p = AD / f'{o}.obj'
    if not p.exists(): p = AD / f'{o}_visual.obj'
    V = np.asarray(trimesh.load(p, force='mesh', process=False).vertices)
    lz = lambda T: np.array([(V @ x[:3, :3].T + x[:3, 3])[:, 2].min() for x in T]) * 100
    a, b = lz(old[o]), lz(G0 @ Tc[o])
    print(o, 'old', [round(float(a[s]), 1) for s in range(0, len(a), 20)]); print(o, 'new', [round(float(b[s]), 1) for s in range(0, len(b), 20)])
    if ach: c = lz(ach[o]); print(o, 'ach', [round(float(c[s]), 1) for s in range(0, len(c), 20)])
