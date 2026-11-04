"""absolute spoon roll (twist about the long axis, unwrapped, relative to frame 0) of target vs achieved, plus spoon height."""
import sys, numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, str(Path(__file__).parent)); import plib as B
ep, actor, md = int(sys.argv[1]), sys.argv[2], sys.argv[3]; every = int(sys.argv[4]) if len(sys.argv) > 4 else 5
tr = np.load(Path(actor) / 'evaluation/execution/trajectory.npz'); on = [str(s) for s in tr['object_names']]; P = {}
for i, n in enumerate(on):
    T = np.tile(np.eye(4), (len(tr['qpos']), 1, 1)); T[:, :3, :3] = R.from_quat(tr['object_quaternions_wxyz'][:, i], scalar_first=True).as_matrix(); T[:, :3, 3] = tr['object_positions'][:, i]; P[n] = T
reg = B.register(ep, {n: Path(md) / f'{n}.obj' for n in B.NAMES}); e, ach, ref = B.per_frame_full(ep, P, reg)
g = B.geometry(ep)['wooden_spoon']; v = g['v'] - g['v'].mean(0); w, U = np.linalg.eigh(v.T @ v); ax = U[:, -1]
s = 1
def twist(Q):
    Rr = R.from_quat(Q); rel = (Rr[0].inv() * Rr).as_quat(); t = 2 * np.arctan2(rel[:, :3] @ ax, rel[:, 3]); return np.degrees(np.unwrap(t))
ta, tt = twist(ach[:, s, 3:]), twist(ref[:, s, 3:])
print(' t   | twist ach | twist tgt | diff | z ach | z tgt | pos err | deg err')
for k in range(0, len(ta), every):
    print(f'{k/20:5.2f} | {ta[k]:8.1f} | {tt[k]:8.1f} | {ta[k]-tt[k]:6.1f} | {ach[k,s,2]*100:6.1f} | {ref[k,s,2]*100:6.1f} | {e["wooden_spoon"][0][k]:5.2f} | {e["wooden_spoon"][1][k]:5.1f}')
