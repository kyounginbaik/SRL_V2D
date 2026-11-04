"""Score a raw physics run (RUN_DIR/result.json -> execution trajectory) vs target_final_v5 like score_any.sh: v21 package rows of the
episode (same meshes / provenance) with pos/quat replaced by the run's object body poses. usage: score_run.py EP RUN_DIR OUT_DIR"""
import sys, json, numpy as np, pandas as pd
from pathlib import Path
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, str(Path(__file__).resolve().parent)); import nlib as L
ep, run, out = int(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]); (out / 'results').mkdir(parents=True, exist_ok=True)
r = json.loads((run / 'result.json').read_text()); tr = np.load(r['execution']['trajectory']); nm = [str(x) for x in tr['object_names']]
s = pd.read_parquet(L.V21); e = s[s.row_id.str.contains(f'/e{ep:06d}/')].copy()
f = e.row_id.str.extract(r'/f(\d+)/o(\d+)').astype(int).values
names = L.NAMES[ep]; n = len(tr['object_positions'])
for k, (fr, slot) in enumerate(f):
    i = nm.index(names[slot]); fr = min(fr, n - 1); p = tr['object_positions'][fr, i]; q = tr['object_quaternions_wxyz'][fr, i]
    e.iloc[k, e.columns.get_loc('pos_x')], e.iloc[k, e.columns.get_loc('pos_y')], e.iloc[k, e.columns.get_loc('pos_z')] = p
    for a, v in zip('xyzw', [q[1], q[2], q[3], q[0]]): e.iloc[k, e.columns.get_loc('quat_' + a)] = v
e.to_parquet(out / 'submission.parquet', index=False)
pd.DataFrame(dict(object_id=list(range(len(names))), object_body_name=names, environment_id=0, step=0)).to_parquet(out / f'results/episode_{ep:06d}.parquet', index=False)
