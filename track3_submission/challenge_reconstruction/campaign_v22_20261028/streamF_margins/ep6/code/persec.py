"""per-second errors of a run's achieved objects vs its episode reference (internal proxy, body origin) + spoon z.
usage: persec.py RUN_DIR [RUN_DIR...]"""
import sys, json, numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
for d in sys.argv[1:]:
    r = json.loads((Path(d) / 'result.json').read_text()); tr = np.load(r['execution']['trajectory'])
    ref = np.load(Path(r['input_dir']) / 'reference_trajectory.npz'); n = min(len(tr['object_positions']), len(ref['object_positions']))
    print(Path(d).name)
    for i, o in enumerate(tr['object_names']):
        j = [str(x) for x in ref['object_names']].index(str(o)) if 'object_names' in ref else i
        dp = np.linalg.norm(tr['object_positions'][:n, i] - ref['object_positions'][:n, j], axis=1) * 100
        ra = R.from_quat(tr['object_quaternions_wxyz'][:n, i][:, [1, 2, 3, 0]]); rb = R.from_quat(ref['object_quaternions_wxyz'][:n, j][:, [1, 2, 3, 0]])
        dr = np.degrees((ra.inv() * rb).magnitude())
        print(' %-13s cm ' % o + ' '.join('%5.1f' % dp[s:s + 20].mean() for s in range(0, n, 20)))
        print(' %-13s dg ' % o + ' '.join('%5.0f' % dr[s:s + 20].mean() for s in range(0, n, 20)))
