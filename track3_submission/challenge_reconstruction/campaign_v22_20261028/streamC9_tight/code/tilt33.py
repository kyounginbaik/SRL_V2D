import sys, numpy as np
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamC9_tight/code')
import c9lib as L
from scipy.spatial.transform import Rotation as R
ep = int(sys.argv[1]); obj = sys.argv[2]; Q = np.load(sys.argv[3])['Q']
import json, mujoco
tg = L.target(ep)[obj]
# relative rotation from frame 0, angle of body z axis change (target) vs physics free joint
m = mujoco.MjModel.from_xml_path(sys.argv[4]); a = int(m.joint(f'{obj}_free').qposadr[0])
Rp = R.from_quat(Q[:, a + 3:a + 7], scalar_first=True); Rt = R.from_matrix(tg[:len(Q), :3, :3])
up = np.array([0, 0, 1.0])
for k in range(0, len(Q), 20):
    # tilt of object relative to its frame-0 orientation, measured as angle of world-up seen in body
    dt = (Rt[k] * Rt[0].inv()).magnitude(); dp = (Rp[k] * Rp[0].inv()).magnitude()
    zt = np.degrees(np.arccos(np.clip(Rt[k].inv().apply(up) @ Rt[0].inv().apply(up), -1, 1)))
    zp = np.degrees(np.arccos(np.clip(Rp[k].inv().apply(up) @ Rp[0].inv().apply(up), -1, 1)))
    print(f'{k/20:5.1f}s target rot {np.degrees(dt):6.1f} tilt-change {zt:6.1f} | phys rot {np.degrees(dp):6.1f} tilt-change {zp:6.1f}  z t {tg[k,2,3]-tg[0,2,3]:+.3f} p {Q[k,a+2]-Q[0,a+2]:+.3f}')
