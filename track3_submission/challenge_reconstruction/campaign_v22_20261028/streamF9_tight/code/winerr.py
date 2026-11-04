"""F9: per-0.5 s window errors (pos cm / deg per object, MPPE) of a residual table on the shipped actor (full official-equivalent rollout).
usage: winerr.py EP ACTOR MESHES TABLE.npz|none"""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mresF9 as mres
ep, actor, meshes, tb = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
env = mres.Env(dict(ep=ep, actor=str(Path(actor).resolve()), meshes=meshes, table=None if tb == 'none' else tb))
r = env.evaluate(env.table0, keep=True); e = mres.L.per_frame(ep, env.poses(r['Q']), env.reg, env.tgt, want_mppe=True)
print(json.dumps({n: [round(float(e[n][0].mean()), 3), round(float(e[n][1].mean()), 2)] for n in env.names}), 'mppe', round(float(e['_mppe'].mean()), 3))
for s0 in range(0, env.T, 10):
    print(f'{s0/env.fps:5.1f}s', ' '.join(f"{n[:6]} {e[n][0][s0:s0+10].mean():5.2f}/{e[n][1][s0:s0+10].mean():5.1f}" for n in env.names), f"mppe {e['_mppe'][s0:s0+10].mean():.2f}")
np.savez_compressed(Path(tb).with_suffix('.Q.npz') if tb != 'none' else 'Q_base.npz', Q=r['Q'])
