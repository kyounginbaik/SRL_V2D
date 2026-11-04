"""Compare target (our body, G0-mapped onto the physics anchor f0) with the current physics reference of an episode dir: per-second
shift; also relative pose between objects at f0. usage: diag_ref.py EP EPISODE_DIR"""
import sys, json, numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, str(Path(__file__).resolve().parent)); import dlib as L
ep = int(sys.argv[1]); epd = Path(sys.argv[2]); names = L.NAMES[ep]
t = np.load(epd / 'tracking_reference.npz', allow_pickle=False); jc = [str(x) for x in t['joint_names']]
def col(o): return jc.index(f'{o}_free[0]') if f'{o}_free[0]' in jc else jc.index(f'{o}_free')
old = {}
for o in names:
    c = col(o); q = t['qpos']; T = np.tile(np.eye(4), (len(q), 1, 1)); T[:, :3, 3] = q[:, c:c + 3]; T[:, :3, :3] = R.from_quat(q[:, c + 3:c + 7], scalar_first=True).as_matrix(); old[o] = T
Tc, X = L.target_ourbody(ep); n = min(len(old[names[0]]), len(Tc[names[0]])); print('frames', len(old[names[0]]), len(Tc[names[0]]))
for a in names:
    G0 = old[a][0] @ np.linalg.inv(Tc[a][0]); print('== G0 anchored on', a, 'rot deg', np.degrees(R.from_matrix(G0[:3, :3]).magnitude()).round(1))
    for o in names:
        new = G0 @ Tc[o][:n]; dp, dr = L.errs(new, old[o][:n])
        print(f'  {o:14s} mean {dp.mean():5.2f} cm {dr.mean():5.1f} deg | per-sec cm', [round(float(dp[s:s+20].mean()), 1) for s in range(0, n, 20)], 'deg', [round(float(dr[s:s+20].mean())) for s in range(0, n, 20)])
        print(f'  {"":14s} f0 dpos(cm) {((new[0,:3,3]-old[o][0,:3,3])*100).round(2)} f0 z new/old {new[0,2,3]*100:.2f}/{old[o][0,2,3]*100:.2f}')
