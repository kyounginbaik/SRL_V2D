"""P8: roll out base + given candidates (gridP term lists), save Q, print in-hand spoon pose (wrist frame) difference vs base per second.
usage: inhand.py ACTOR OUTDIR 'JSON list of term-lists'"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gridP, ilcK
from scipy.spatial.transform import Rotation as R
actor, out, cands = sys.argv[1], Path(sys.argv[2]), json.loads(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
cfg = dict(ep=35, actor=str(Path(actor).resolve()), table=None, epd=None, failp=99.0, faild=999.0, need_base=True); gridP.init(cfg); W = gridP.W
def inhand(Q):
    sc = W.sc; Rw = W.Ac * R.from_euler('XYZ', Q[:, W.wq[3:]]); pw = Q[:, W.wq[:3]]
    Rs = R.from_quat(Q[:, sc + 3:sc + 7], scalar_first=True); ps = Q[:, sc:sc + 3]
    return Rw.inv() * Rs, Rw.inv().apply(ps - pw)
Qb = gridP.BASEQ; np.save(out / 'Q_base.npy', Qb); Rb, pb = inhand(Qb)
for i, terms in enumerate(cands):
    Q = W.rollout(gridP.table_of(W, W.table0, terms)); np.save(out / f'Q_{i}.npy', Q); Rc, pc = inhand(Q)
    print('cand', i, json.dumps(terms), 'score', W.score(Q)[1])
    dr = (Rb.inv() * Rc).as_rotvec(degrees=True); dp = (pc - pb) * 100
    for s in range(0, W.T, 20): print(f'  {s/20:5.1f}s inhand drot {np.round(dr[s],0)} |{np.linalg.norm(dr[s]):5.1f}| dpos {np.round(dp[s],1)}')
