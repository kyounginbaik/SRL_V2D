"""signed spoon/pot position error (aligned scorer frame, achieved - target), per second, plus spoon-anchor distance to pot anchor."""
import sys, numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, str(Path(__file__).parent)); import plib as B
ep, actor, md = int(sys.argv[1]), sys.argv[2], sys.argv[3]
tr = np.load(Path(actor) / 'evaluation/execution/trajectory.npz'); on = [str(s) for s in tr['object_names']]; P = {}
for i, n in enumerate(on):
    T = np.tile(np.eye(4), (len(tr['qpos']), 1, 1)); T[:, :3, :3] = R.from_quat(tr['object_quaternions_wxyz'][:, i], scalar_first=True).as_matrix(); T[:, :3, 3] = tr['object_positions'][:, i]; P[n] = T
reg = B.register(ep, {n: Path(md) / f'{n}.obj' for n in B.NAMES}); e, ach, ref = B.per_frame_full(ep, P, reg)
for k in range(0, len(ach), 20):
    sl = slice(k, k + 20); ds = (ach[sl, 1, :3] - ref[sl, 1, :3]).mean(0) * 100; dp = (ach[sl, 0, :3] - ref[sl, 0, :3]).mean(0) * 100
    rel_a = np.linalg.norm(ach[sl, 1, :3] - ach[sl, 0, :3], axis=1).mean() * 100; rel_t = np.linalg.norm(ref[sl, 1, :3] - ref[sl, 0, :3], axis=1).mean() * 100
    print(f'{k/20:4.0f}s spoon d={np.round(ds,1)} pot d={np.round(dp,1)} |spoon-pot| ach {rel_a:5.1f} tgt {rel_t:5.1f}')
