"""usage: refdelta.py OLD_EPD NEW_EPD : per-second spoon delta new vs old reference (both setup frame): pos cm, angle deg, body-frame rotvec axis"""
import sys, numpy as np
from scipy.spatial.transform import Rotation as R
qo = np.load(sys.argv[1] + '/tracking_reference.npz')['qpos']; qn = np.load(sys.argv[2] + '/tracking_reference.npz')['qpos']
for name, c in (('cup', 56), ('spoon', 63)):
    So = R.from_quat(qo[:, c + 3:c + 7], scalar_first=True); Sn = R.from_quat(qn[:, c + 3:c + 7], scalar_first=True)
    rv = (So.inv() * Sn).as_rotvec(); dp = np.linalg.norm(qn[:, c:c + 3] - qo[:, c:c + 3], axis=1) * 100
    print(name)
    for s in range(0, len(qo), 20):
        sl = slice(s, s + 20); a = np.degrees(np.linalg.norm(rv[sl], axis=1)); ax = rv[sl].mean(0); ax = ax / (np.linalg.norm(ax) + 1e-9)
        print(f'  {s/20:4.0f}s pos {dp[sl].mean():5.2f} ang {a.mean():6.1f} (max {a.max():6.1f}) axis {np.round(ax, 2)}')
