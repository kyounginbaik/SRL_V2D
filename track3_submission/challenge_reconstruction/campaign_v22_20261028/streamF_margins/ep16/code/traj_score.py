"""tlib16 scorer-identical errors of execution trajectory.npz files (object free-joint qpos)."""
import sys, numpy as np, mujoco; sys.path.insert(0, sys.path[0]); import tlib16 as L
from scipy.spatial.transform import Rotation as R
m = mujoco.MjModel.from_xml_path('/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamN_ep16/world_6prong_newtrack/scene/controller_scene.xml')
for f in sys.argv[1:]:
    z = np.load(f); Q = z['qpos']; Ta = {}
    for o in L.NAMES:
        c = int(m.joint(o + '_free').qposadr[0]); T = np.tile(np.eye(4), (len(Q), 1, 1)); T[:, :3, 3] = Q[:, c:c + 3]; T[:, :3, :3] = R.from_quat(Q[:, c + 3:c + 7][:, [1, 2, 3, 0]]).as_matrix(); Ta[o] = T
    e = L.errors(Ta); print(f.split('/')[-3], {o: (round(float(p.mean()), 2), round(float(d.mean()), 1)) for o, (p, d) in e.items()}, 'frames', len(Q))
