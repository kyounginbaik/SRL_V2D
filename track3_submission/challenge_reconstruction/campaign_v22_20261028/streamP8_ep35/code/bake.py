"""P8: bake a gridP candidate (right-wrist roll / finger / LEFT-wrist offsets) into a new actor checkpoint (phase weights only; feedback,
margins, scene unchanged) and run the official evaluate_policy_component on it (via ilcK.save_actor).
usage: bake.py ACTOR OUT 'JSON term list'"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import sys, json
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gridP, ilcK
actor, out, terms = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
cfg = dict(ep=35, actor=str(Path(actor).resolve()), table=None, epd=None, failp=99.0, faild=999.0, need_base=any('roll' in t for t in terms)); gridP.init(cfg); W = gridP.W
tb = gridP.table_of(W, W.table0, terms); Q = W.rollout(tb); print('search', W.score(Q)[1], flush=True)
ilcK.save_actor(W, tb, out, dict(mode='P8 bake', terms=terms, base_actor=cfg['actor'], search=W.score(Q)[1], baker=str(Path(__file__).resolve())))
