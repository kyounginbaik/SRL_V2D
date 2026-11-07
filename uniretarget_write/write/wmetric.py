"""Writing score, fixed before any method ran. PASS: median tip error while writing under 1 cm, tip on the board for at least 80% of writing frames, pen still in hand at the end."""
import json, numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
H = Path('/scratch2/exaFLOPs/srl_yam_run/ablate/oak/write'); P = Path('/scratch2/exaFLOPs/spider_wide/ds/processed/oakinkv2/sharpa/right/write_wb')
B = json.load(open(H / 'board.json')); N = np.array(B['normal']); S = np.array(B['surface_point']); TL = np.array(B['tip_local'])
def tip_of(pen):  # pen: (n,7) xyz + wxyz
    return R.from_quat(pen[:, 3:][:, [1, 2, 3, 0]]).apply(TL) + pen[:, :3]
def ref():
    """Goal = raw mocap pen in the scene frame. SPIDER's IK drops 4 frames and smooths the pen slightly; find the 2-frame shift from position and check it."""
    k = np.load(P / '0/trajectory_kinematic.npz')['qpos']; t = np.load(H / 'tip_ref.npz'); oc = np.load(str(P / '0/trajectory_kinematic.npz').replace('sharpa', 'mano').replace('trajectory_kinematic', 'trajectory_keypoints'))['qpos_obj_right']
    n = len(k); sh = int(np.argmin([np.abs(k[:, -7:-4] - oc[i:i + n, :3]).max() for i in range(len(oc) - n + 1)])); assert np.abs(k[:, -7:-4] - oc[sh:sh + n, :3]).max() < .002, 'IK pen position off the mocap'
    pen = oc[sh:sh + n]; tip = tip_of(pen); assert np.abs(tip - t['tip'][sh:sh + n]).max() < 1e-3, 'tip_ref mismatch'
    return dict(q=k, pen=pen, tip=tip, h=(tip - S) @ N, on=t['on_board'][sh:sh + n].astype(bool), n=n, shift=sh)
RF = ref(); H0 = float(np.median(RF['h'][RF['on']]))
def score(pen, name=''):
    """pen: (n,7) at 50 Hz, same clock as the reference."""
    n = min(len(pen), RF['n']); pen = pen[:n]; on = RF['on'][:n]; rt = RF['tip'][:n]; t = tip_of(pen); h = (t - S) @ N
    te = np.linalg.norm(t - rt, axis=1); pe = np.linalg.norm(pen[:, :3] - RF['pen'][:n, :3], axis=1)
    re = (R.from_quat(pen[:, 3:][:, [1, 2, 3, 0]]) * R.from_quat(RF['pen'][:n, 3:][:, [1, 2, 3, 0]]).inv()).magnitude()
    ob = np.abs(h - H0) < .005; held = bool(pe[-1] < .10)
    tm, of = float(np.median(te[on])), float(ob[on].mean())
    return dict(name=name, frames=int(n), tip_med_cm=round(tm * 100, 2), tip_p90_cm=round(float(np.percentile(te[on], 90)) * 100, 2), on_board=round(of, 3), below_board_cm=round(float(-(h - H0).min()) * 100, 2), pen_cm=round(float(pe.mean()) * 100, 2), pen_rad=round(float(re.mean()), 3), end_cm=round(float(pe[-1]) * 100, 1), held=held, PASS=bool(tm < .01 and of >= .8 and held and n == RF['n']))
if __name__ == '__main__':
    print('ref frames', RF['n'], 'writing frames', int(RF['on'].sum()), 'H0 cm', round(H0 * 100, 3)); print(score(RF['pen'], 'reference'))
