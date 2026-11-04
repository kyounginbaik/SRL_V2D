"""F9: compare the mresF9 base_eval rollout qpos with the build-v2 re-execution qpos of the current tag (bit-exact check)."""
import sys, numpy as np
from pathlib import Path
q = np.load(sys.argv[1])['Q']; tr = next(Path(sys.argv[2]).glob('backend/evaluation/execution/trajectory.npz'))
b = np.load(tr)['qpos']; n = min(len(q), len(b))
print(sys.argv[2].split('/')[-2], q.shape, b.shape, 'max|dq|', float(np.abs(q[:n] - b[:n]).max()), 'exact', bool(np.array_equal(q[:n], b[:n])))
