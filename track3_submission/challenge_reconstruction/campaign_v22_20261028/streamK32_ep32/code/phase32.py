"""K32: per-phase kit-faithful errors (h13lib.per_frame vs $E31_TGT) of saved actors' official rollouts. usage: phase32.py ACTOR[,ACTOR..] MESHES"""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mresH as R
M = sys.argv[2]
for A in sys.argv[1].split(','):
    env = R.Env(dict(ep=32, actor=A, meshes=M)); rep = env.evaluate(env.table0, keep=True)
    e = R.L.per_frame(32, env.poses(rep['Q']), env.reg, env.tgt)
    segs = [(0, 30), (30, 70), (70, 100), (100, 130), (130, 160), (160, 190), (190, 259)]
    print('ACTOR', A, json.dumps(rep['objects']))
    for n in env.names:
        p, d = np.asarray(e[n][0]), np.asarray(e[n][1])
        print('  ', n, ' '.join(f'f{s}-{t}:{p[s:t].mean():.2f}/{d[s:t].mean():.0f}' for s, t in segs), flush=True)
        np.save(Path(__file__).resolve().parents[1] / 'out' / f'perframe_{Path(A).parents[0].name}_{n}.npy', np.stack([p, d]))
