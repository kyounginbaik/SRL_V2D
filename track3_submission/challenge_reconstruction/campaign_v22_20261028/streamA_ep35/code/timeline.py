"""Per-window object error vs the run's episode reference (setup frame, body origin): usage timeline.py RUN_DIR [STEP]"""
import sys, json, numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
d = Path(sys.argv[1]); st = int(sys.argv[2]) if len(sys.argv) > 2 else 20
r = json.loads((d / 'result.json').read_text()); tr = np.load(r['execution']['trajectory'])
epd = Path(r.get('input_dir') or r['evaluation_input_dir']); ref = np.load(epd / 'reference_trajectory.npz')
names = [str(x) for x in tr['object_names']]; n = min(len(tr['object_positions']), len(ref['object_positions']))
for s in range(0, n, st):
    e = min(n, s + st); row = []
    for i, o in enumerate(names):
        dp = np.linalg.norm(tr['object_positions'][s:e, i] - ref['object_positions'][s:e, i], axis=1).mean() * 100
        ra = R.from_quat(tr['object_quaternions_wxyz'][s:e, i][:, [1, 2, 3, 0]]); rb = R.from_quat(ref['object_quaternions_wxyz'][s:e, i][:, [1, 2, 3, 0]])
        row.append(f'{o[:5]} {dp:6.2f} {np.degrees((ra * rb.inv()).magnitude()).mean():5.1f} z {tr["object_positions"][s:e, i, 2].mean()*100:5.1f}/{ref["object_positions"][s:e, i, 2].mean()*100:5.1f}')
    print(f'f{s:3d}-{e-1:3d} ' + ' | '.join(row))
