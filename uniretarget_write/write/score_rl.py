"""Score every saved checkpoint of the whiteboard RL runs on the writing rule, over all recorded worlds. Policy frame j is time (j + 1) / fps, with fps taken from the run itself."""
import sys, json, glob, re, numpy as np, wmetric as W
from pathlib import Path
from scipy.spatial.transform import Rotation as R, Slerp
RF = W.RF; n = RF['n']; rows = []
for f in sorted(glob.glob(str(W.H.parent / 'runs/write_*/*-s4[234]/eval_*.npz'))):
    d = np.load(f); o = d['o']; T, Wn = o.shape[:2]; fps = round((T + 1) / 11.28); tp = (np.arange(T) + 1) / fps; ti = (np.arange(n) + RF['shift']) * .02; ti = ti[(ti >= tp[0]) & (ti <= tp[-1])]; offset = int(round(tp[0] / .02)) - RF['shift'] if False else None
    res = []
    for w in range(Wn):
        b = o[:, w, 0]; q = b[:, 3:7].astype(float); bad = ~np.isfinite(b).all(1) | (np.linalg.norm(np.nan_to_num(q), axis=1) < 1e-6)
        if bad.any(): res.append(dict(PASS=False, tip_med_cm=99., on_board=0., held=False, pen_cm=99.)); continue
        rr = R.from_quat(q); i0 = np.searchsorted((np.arange(n) + RF['shift']) * .02, tp[0]); pen = np.c_[np.c_[[np.interp(ti, tp, b[:, x]) for x in range(3)]].T, Slerp(tp, rr)(ti).as_quat()[:, [3, 0, 1, 2]]]
        full = np.tile(pen[0], (n, 1)); full[i0:i0 + len(pen)] = pen; full[i0 + len(pen):] = pen[-1]; res.append(W.score(full))
    p = Path(f); step = int(re.search(r'eval_(\d+)', p.name).group(1)); arm = p.parent.parent.name + '/' + p.parent.name
    rows.append(dict(arm=arm, step=step, fps=fps, worlds=Wn, passed=float(np.mean([r['PASS'] for r in res])), tip=float(np.median([r['tip_med_cm'] for r in res])), on=float(np.median([r['on_board'] for r in res])), held=float(np.mean([r['held'] for r in res])), pen=float(np.median([r['pen_cm'] for r in res])))); print(rows[-1], flush=True)
json.dump(rows, open(W.H / 'practice_scores.json', 'w'), indent=0)
