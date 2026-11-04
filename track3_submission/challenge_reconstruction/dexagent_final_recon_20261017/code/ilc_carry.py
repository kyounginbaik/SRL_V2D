"""One ILC step for the carry correction: measure the residual pot pose error of a previous wrapper rollout
and write new time-table parameters (previous correction minus gain x smoothed residual).
  ilc_carry.py --previous <evaluation dir with ../params.json> --out <new variant dir> [--gain-rot 0.6 --gain-pos 0.5 --sigma-s 0.3]
     [--finger-bias RAD]  (closing bias on the holding hand's fingers during the window; sign fitted from the teacher)
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R
ROOT = Path(__file__).resolve().parents[1]

def smooth(x, sigma_frames):
    return gaussian_filter1d(x, sigma_frames, axis=0, mode='nearest') if sigma_frames > 0 else x

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--previous', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--gain-rot', type=float, default=0.6); ap.add_argument('--gain-pos', type=float, default=0.5); ap.add_argument('--sigma-s', type=float, default=0.3)
    ap.add_argument('--finger-bias', type=float, default=None); ap.add_argument('--max-rot-deg', type=float, default=60.); ap.add_argument('--max-pos-m', type=float, default=0.08)
    a = ap.parse_args(); prev = Path(a.previous).resolve(); params = json.load(open(prev.parent / 'params.json')); res = json.load(open(prev / 'result.json'))
    e = int(res['episode'].split('_')[-1]); tr = np.load(res['execution']['trajectory']); t = tr['timestamps_seconds']
    sub = pd.read_parquet(ROOT / f'refinement_20261018/shared_policy_handoff/episode_{e:06d}/results/episode_{e:06d}.parquet'); p = sub[sub.object_body_name == 'white_pot'].sort_values('step')
    qr = R.from_quat(p[['reference_quaternion_x', 'reference_quaternion_y', 'reference_quaternion_z', 'reference_quaternion_w']].to_numpy()); pr = p[['reference_position_x', 'reference_position_y', 'reference_position_z']].to_numpy()
    i = list(tr['object_names']).index('white_pot'); qa = R.from_quat(tr['object_quaternions_wxyz'][:, i][:, [1, 2, 3, 0]]); pa_sim = tr['object_positions'][:, i]
    # achieved positions in the exported (reference) frame: pure z shift between sim and export frames
    z_shift = pr[0, 2] - pa_sim[0, 2]; pa = pa_sim + np.array([0, 0, z_shift])
    n = min(len(t), len(pr)); sig = a.sigma_s * 20.
    err_rot = smooth((qa[:n] * qr[:n].inv()).as_rotvec(), sig)          # world: reference -> achieved
    err_pos = smooth(pr[:n] - pa[:n], sig)                                # world: achieved -> reference
    if 'table_time_s' in params:
        tt = np.asarray(params['table_time_s']); prev_rot = np.asarray(params['rotation_table']); prev_pos = np.asarray(params['translation_table'])
        prev_rot = np.stack([np.interp(t[:n], tt, prev_rot[:, k]) for k in range(3)], 1); prev_pos = np.stack([np.interp(t[:n], tt, prev_pos[:, k]) for k in range(3)], 1)
    else:
        prev_rot = np.tile(np.asarray(params['rotation_vector']) * params['rot_gain'], (n, 1)); prev_pos = np.tile(np.asarray(params['translation_m']) * params['pos_gain'], (n, 1))
    # compose rotations properly: new = exp(-k*err) * prev (small-angle addition is fine for these magnitudes, but keep exact)
    new_rot = (R.from_rotvec(-a.gain_rot * err_rot) * R.from_rotvec(prev_rot)).as_rotvec()
    new_pos = prev_pos + a.gain_pos * err_pos
    mag = np.linalg.norm(new_rot, axis=1, keepdims=True); cap = np.radians(a.max_rot_deg); new_rot = np.where(mag > cap, new_rot / np.maximum(mag, 1e-9) * cap, new_rot)
    new_pos = np.clip(new_pos, -a.max_pos_m, a.max_pos_m)
    out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    new = {k: v for k, v in params.items() if k not in ('rotation_vector', 'translation_m', 'rot_gain', 'pos_gain', 'table_time_s', 'rotation_table', 'translation_table')}
    new.update(table_time_s=t[:n].tolist(), rotation_table=new_rot.tolist(), translation_table=new_pos.tolist(), ilc_previous=str(prev), ilc_gain_rot=a.gain_rot, ilc_gain_pos=a.gain_pos, ilc_sigma_s=a.sigma_s,
               ilc_iteration=int(params.get('ilc_iteration', 0)) + 1)
    if a.finger_bias is not None:
        names = tr['actuator_names'].tolist(); side = params.get('side', 'left')
        ids = [k for k, nm in enumerate(names) if nm.startswith(f'robot_{side}_{side}_') and not any(s in nm for s in ('_pos_', '_rot_'))]
        ctrl = tr['ctrl']; g0 = ctrl[0, ids]; w = (t >= params['start_s']) & (t <= params['hold_end_s']); sign = np.sign(ctrl[w][:, ids].mean(0) - g0); sign[sign == 0] = 1.
        new.update(finger_actuator_ids=ids, finger_bias_rad=(a.finger_bias * sign).tolist(), finger_bias_magnitude_rad=a.finger_bias)
    json.dump(new, open(out / 'params.json', 'w'), indent=1)
    gate_w = (t[:n] >= params['start_s']) & (t[:n] <= params['hold_end_s'])
    print(json.dumps(dict(episode=e, iteration=new['ilc_iteration'], residual_rot_deg_mean_window=float(np.degrees(np.linalg.norm(err_rot[gate_w], axis=1)).mean()),
                          residual_pos_cm_mean_window=float(100 * np.linalg.norm(err_pos[gate_w], axis=1).mean()), new_rot_deg_max=float(np.degrees(np.linalg.norm(new_rot, axis=1)).max()),
                          new_pos_cm_max=float(100 * np.abs(new_pos).max()), params=str(out / 'params.json'))))

if __name__ == '__main__':
    main()
