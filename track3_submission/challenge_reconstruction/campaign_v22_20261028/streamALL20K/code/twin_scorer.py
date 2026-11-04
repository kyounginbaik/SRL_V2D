"""E31: does the cwmesh best-seed transform (bundle -> component) agree with the scorer's own registration (component -> target mesh =
bundle mesh, scorer seed, budgeted shipped mesh), which v29's passing scores validate? rotation of R_scorer @ R_plan should be ~0.
usage: E31_TGT=target_E14_HGA twin_scorer.py PLAN.json EP:OBJ ..."""
import sys, json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, str(Path(__file__).resolve().parent)); import h13lib as L
plan = json.loads(Path(sys.argv[1]).read_text())
for key in sys.argv[2:]:
    e, n = key.split(':'); ep = int(e); p = plan[key]
    if ep not in L.NAMES: L.NAMES[ep] = [k.split(':')[1] for k in plan if int(k.split(':')[0]) == ep]
    reg = L.register(ep, {m: Path(plan[f'{e}:{m}']['component_mesh']) for m in L.NAMES[ep]})
    ca, Rs = reg[n]; Rp = np.asarray(p['R'])
    g = L.geometry(ep)[n]; ax = np.asarray(g['axis'], float); sym = np.linalg.norm(ax) > 0 and g['order'] != 1
    Rc = Rs @ Rp
    gap = np.degrees(np.arccos(np.clip(np.dot(Rc @ (Rp.T @ Rs.T @ ax) if False else Rc.T @ ax, ax), -1, 1))) if sym else np.degrees(Rot.from_matrix(Rc).magnitude())
    print(key, 'scorer-vs-plan rotation', round(float(gap), 2), 'deg', 'sym' if sym else '', flush=True)
