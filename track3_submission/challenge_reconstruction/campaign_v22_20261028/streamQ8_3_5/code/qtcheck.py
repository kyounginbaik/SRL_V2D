"""score the tracking reference qt itself (objects) with frame 0 = physics start Q[0] vs qt[0]: shows the systematic error of
perfect qt-tracking under the scorer's frame-0 alignment."""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamL8_3_5_6/code')
import l8res as L, plib as B
ep = int(sys.argv[1]); env = L.Env(dict(ep=ep, actor=sys.argv[2], meshdir=sys.argv[3]))
Q = np.load(Path(sys.argv[2]) / 'evaluation/execution/trajectory.npz')['qpos']
qt = env.qt[:len(Q)].copy()
for lab, QQ in [('actor', Q), ('qt', qt), ('qt_with_phys_f0', np.vstack([Q[:1], qt[1:]]))]:
    e = B.per_frame(ep, env.poses(QQ), env.reg, env.tgt)
    print(lab, {n: (round(float(e[n][0].mean()), 3), round(float(e[n][1].mean()), 2)) for n in B.NAMES})
sc = env.sc; from scipy.spatial.transform import Rotation as R
print('frame0 spoon phys vs qt: dpos cm', np.round((Q[0, sc:sc+3] - qt[0, sc:sc+3]) * 100, 2), 'drot deg', np.degrees((R.from_quat(Q[0, sc+3:sc+7], scalar_first=True) * R.from_quat(qt[0, sc+3:sc+7], scalar_first=True).inv()).magnitude()).round(2))
pc = env.adr['white_pot']; print('frame0 pot phys vs qt dpos cm', np.round((Q[0, pc:pc+3] - qt[0, pc:pc+3]) * 100, 2))
