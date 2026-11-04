"""where is the stirring amplitude lost? hp (2 Hz band) gains: target-implied wrist (rigid hold) vs commanded wrist ctrl vs achieved wrist vs spoon."""
import sys, json, numpy as np, mujoco
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import gaussian_filter1d as gf
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamL8_3_5_6/code')
import l8res as L
ep = int(sys.argv[1]); env = L.Env(dict(ep=ep, actor=sys.argv[2], meshdir=sys.argv[3])); t0, t1 = float(sys.argv[4]), float(sys.argv[5])
z = np.load(Path(sys.argv[2]) / 'evaluation/execution/trajectory.npz'); Q = z['qpos']; C = z['ctrl']; names = [str(s) for s in z['actuator_names']]
ia = [names.index(f'robot_right_right_{j}_position') for j in ('pos_x', 'pos_y', 'pos_z')]
pw = Q[:, env.wq[:3]]; cw = C[:, ia]; sc = env.sc; ps = Q[:, sc:sc + 3]; pt = env.qt[:len(Q), sc:sc + 3]
corr = env.ilc_correction(Q, 0.0, 1e9)  # rigid-hold implied wrist correction (no window)
iw = pw + corr[:, :3]
k = slice(int(t0 * 20), int(t1 * 20))
def hp(x, s=4): return (x - gf(x, s, axis=0, mode='nearest'))[k]
ref = hp(pt)
for lab, x in [('spoon target', pt), ('spoon achieved', ps), ('wrist implied by target (rigid)', iw), ('wrist commanded ctrl', cw), ('wrist achieved', pw)]:
    h = hp(x); g = (h * ref).sum() / (ref * ref).sum()
    # lag vs target in frames by xcorr
    lags = range(-4, 5); cc = [((np.roll(hp(x), l, 0)) * ref).sum() for l in lags]
    print(f'{lab:34s} rms_hp {np.sqrt((h**2).sum(1).mean())*100:5.2f} cm  gain_vs_target {g:5.2f}  best_lag_frames {list(lags)[int(np.argmax(cc))]}')
h1 = hp(cw); h2 = hp(pw); print('achieved wrist / commanded hp gain', round(float((h1 * h2).sum() / (h1 * h1).sum()), 3))
h1 = hp(pw); h2 = hp(ps); print('spoon / achieved wrist hp gain', round(float((h1 * h2).sum() / (h1 * h1).sum()), 3))
