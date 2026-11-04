"""Rank variant evaluations (result.json under a glob) by pot error, tilt and sag against the reference."""
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial.transform import Rotation as R
root = Path(__file__).resolve().parents[1]


def analyse(result_path):
    res = json.load(open(result_path)); e = int(res['episode'].split('_')[-1])
    sub = pd.read_parquet(root / f'refinement_20261018/shared_policy_handoff/episode_{e:06d}/results/episode_{e:06d}.parquet')
    p = sub[sub.object_body_name == 'white_pot'].sort_values('step')
    qr = R.from_quat(p[['reference_quaternion_x', 'reference_quaternion_y', 'reference_quaternion_z', 'reference_quaternion_w']].to_numpy())
    pr = p[['reference_position_x', 'reference_position_y', 'reference_position_z']].to_numpy()
    lifted = pr[:, 2] > pr[0, 2] + 0.01; up = qr[0].inv().apply([0, 0, 1])
    m = res['metrics']['official_style_reference_proxy']; tr = np.load(res['execution']['trajectory'])
    n = len(tr['timestamps_seconds']); i = list(tr['object_names']).index('white_pot')
    qa = R.from_quat(tr['object_quaternions_wxyz'][:n, i][:, [1, 2, 3, 0]]); pa = tr['object_positions'][:n, i]
    tilt = np.degrees(np.arccos(np.clip(np.sum(qa.apply(up) * qr[:n].apply(up), 1), -1, 1)))
    dz = (pa[:, 2] - pa[0, 2]) - (pr[:n, 2] - pr[0, 2]); L = lifted[:n]
    objs = {k: (round(v['mean_position_error_cm'][0], 2), round(v['mean_orientation_error_degrees'][0], 1)) for k, v in m['objects'].items()}
    return dict(ep=e, variant=str(Path(result_path).parent.parent.name), status='PASS' if m['mp_sr'] == 1 else 'fail',
                pot=objs['white_pot'], other={k: v for k, v in objs.items() if k != 'white_pot'},
                tilt_lifted_deg=round(float(tilt[L].mean()), 1), tilt_max_deg=round(float(tilt.max()), 1),
                sag_lifted_cm=round(float(dz[L].mean() * 100), 2), completed=res['execution']['execution_completed'],
                frames=n, replay_ok=res['execution']['independent_control_replay']['ok'], result=str(result_path))


if __name__ == '__main__':
    rows = [analyse(r) for r in sorted(glob.glob(sys.argv[1]))]
    rows.sort(key=lambda r: (r['ep'], r['status'] != 'PASS', r['pot'][1] + 10 * r['pot'][0]))
    for r in rows:
        print(f"ep{r['ep']} {r['status']:4s} {r['variant']:34s} pot {r['pot'][0]:5.2f}cm {r['pot'][1]:5.1f}deg  other {r['other']}  tilt {r['tilt_lifted_deg']:5.1f} (max {r['tilt_max_deg']:5.1f}) sag {r['sag_lifted_cm']:+5.2f}cm  frames {r['frames']}")
    if len(sys.argv) > 2: json.dump(rows, open(sys.argv[2], 'w'), indent=1)
