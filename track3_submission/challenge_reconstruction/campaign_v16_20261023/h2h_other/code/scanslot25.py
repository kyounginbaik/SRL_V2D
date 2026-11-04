"""ep25: the scan's slot direction (scan body +y, from slot.py on gt_best_per_object) carried into the camera through each
track's kit registration (exact scorer seed), vs the slot direction the video supports (NEW's own mesh slot, verified on the
RGB at f0/f160, ep25_slot_f*.jpg). Tells whether the official registration of each track's mesh puts the scan slot on the right side."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(__file__)); from common import *
import h2h
o = load_ours(25); n = load_new(25, o['names']); res = {}
for k in o['names']:
    truth = n['A_T_obj'][k][:, :3, :3] @ np.array([0, 0, 1.])
    r = {}
    for tn in ('ours', 'new'):
        REG = h2h.registration((o['meshes'] if tn == 'ours' else n['meshes'])[k], k, 25, o['names'].index(k))
        A = o['A_T_obj'][:, o['names'].index(k)] if tn == 'ours' else n['A_T_obj'][k]
        d = np.einsum('tij,jk,k->ti', A[:, :3, :3], REG[:3, :3], np.array([0, 1., 0]))
        ang = np.degrees(np.arccos(np.clip(np.sum(d * truth, 1), -1, 1))); r[tn] = dict(median_deg=float(np.median(ang)), min=float(ang.min()), max=float(ang.max()))
    ours_own = o['A_T_obj'][:, o['names'].index(k), :3, :3] @ np.array([0, 0, -1.]); a2 = np.degrees(np.arccos(np.clip(np.sum(ours_own * truth, 1), -1, 1)))
    r['ours_collision_slot_vs_video'] = dict(median_deg=float(np.median(a2)), min=float(a2.min()), max=float(a2.max())); res[k] = r; print(k, json.dumps(r))
json.dump(res, open(sys.argv[1], 'w'), indent=1)
