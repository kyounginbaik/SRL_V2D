"""Practice in the Batch: write a clip that holds K copies of one clip end to end, each padded with HOLD frames of the last pose.
Each copy is one strategy slot. At 50 Hz source and 40 Hz control, (n + HOLD) must be a multiple of 5 so each slot is exactly (n + HOLD) * 0.8 control frames.
Usage: tile_clip.py SRC_TASK DST_TASK K"""
import sys, json, numpy as np, pyarrow as pa, pyarrow.parquet as pq
from pathlib import Path
C = Path('/scratch2/exaFLOPs/srl_yam_run/ablate/oak/clips'); src, dst, K = sys.argv[1], sys.argv[2], int(sys.argv[3])
r = pq.read_table(C / src / 'none/reference.parquet').to_pylist()[0]; n = len(r['robot_right_wrist_position'])
HOLD = (-n) % 5 + 25  # half a second of hold, rounded so the slot length is a multiple of 5
L = n + HOLD; SEG = int(round(L * 40 / r['fps'])); assert abs(L * 40 / r['fps'] - SEG) < 1e-9, 'slot does not align with the control rate'
out = dict(r)
for k, v in r.items():
    if isinstance(v, list) and len(v) == n:
        one = v + [v[-1]] * HOLD
        out[k] = one * K + [v[-1]] * 5  # a short tail so resampling never drops the last slot's final frame
(C / dst / 'none').mkdir(parents=True, exist_ok=True)
pq.write_table(pa.Table.from_pylist([out]), C / dst / 'none/reference.parquet')
json.dump(dict(source=src, K=K, source_frames=n, hold=HOLD, slot_source_frames=L, slot_control_frames=SEG, clip_control_frames=int(round((n - 1) * 40 / r['fps'])) + 1), open(C / dst / 'tiling.json', 'w'))
print(json.load(open(C / dst / 'tiling.json')))
