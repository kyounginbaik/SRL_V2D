"""Rank physics pickup/retention search results (result.json with metrics + execution) by pot/other errors and tilt."""
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial.transform import Rotation as R
root = Path(__file__).resolve().parents[1]
rows = []
for r in sorted(glob.glob(sys.argv[1])):
    res = json.load(open(r)); e = int(Path(r).parents[1].name.split('_')[-1]) if 'episode_' in Path(r).parents[1].name else int(Path(r).parents[2].name.split('_')[-1])
    m = res['metrics']['official_style_reference_proxy']; tr = np.load(res['execution']['trajectory'])
    sub = pd.read_parquet(root / f'refinement_20261018/shared_policy_handoff/episode_{e:06d}/results/episode_{e:06d}.parquet'); p = sub[sub.object_body_name == 'white_pot'].sort_values('step')
    qr = R.from_quat(p[['reference_quaternion_x', 'reference_quaternion_y', 'reference_quaternion_z', 'reference_quaternion_w']].to_numpy()); pr = p[['reference_position_x', 'reference_position_y', 'reference_position_z']].to_numpy()
    n = len(tr['timestamps_seconds']); i = list(tr['object_names']).index('white_pot'); qa = R.from_quat(tr['object_quaternions_wxyz'][:n, i][:, [1, 2, 3, 0]]); pa = tr['object_positions'][:n, i]
    up = qr[0].inv().apply([0, 0, 1]); tilt = np.degrees(np.arccos(np.clip(np.sum(qa.apply(up) * qr[:n].apply(up), 1), -1, 1))); L = pr[:n, 2] > pr[0, 2] + 0.01
    dz = (pa[:, 2] - pa[0, 2]) - (pr[:n, 2] - pr[0, 2])
    objs = {k: (round(v['mean_position_error_cm'][0], 2), round(v['mean_orientation_error_degrees'][0], 1)) for k, v in m['objects'].items()}
    best = res.get('best', {})
    rows.append(dict(ep=e, run='/'.join(Path(r).parts[-3:-1]), status='PASS' if m['mp_sr'] == 1 else 'fail', pot=objs.get('white_pot'), other={k: v for k, v in objs.items() if k != 'white_pot'},
                     tilt=round(float(tilt[L].mean()), 1) if L.any() else None, sag=round(float(dz[L].mean() * 100), 2) if L.any() else None, frames=n,
                     completed=res['execution'].get('execution_completed'), search_cost=round(best.get('cost', float('nan')), 2), qualified=round(best.get('longest_qualified_unsupported_contact_s', float('nan')), 2)))
rows.sort(key=lambda r: (r['ep'], r['status'] != 'PASS', (r['pot'] or (99, 99))[0] + 0.1 * (r['pot'] or (99, 99))[1]))
for r in rows: print(f"ep{r['ep']} {r['status']:4s} {r['run']:38s} pot {r['pot']} other {r['other']} tilt {r['tilt']} sag {r['sag']} frames {r['frames']} cost {r['search_cost']} qual {r['qualified']}")
