"""ep19: reproduce the 40 cm / 177 deg dust-pan gap of v20_vs_compose as an anchor-alignment artefact. The kit aligns the
whole scene once by the anchor's (slot 0 = mini_sweeper) frame-0 orientation. Apply that alignment to OUR reference vs NEW
(scan-registered poses, camera frame 0) and report the dust-pan error with and without it."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(__file__)); from common import *
import h2h
from scipy.spatial.transform import Rotation as R
o = load_ours(19); n = load_new(19, o['names']); S = {}
for tn in ('ours', 'new'):
    for k in o['names']:
        A = o['A_T_obj'][:, o['names'].index(k)] if tn == 'ours' else n['A_T_obj'][k]
        S[(tn, k)] = A @ h2h.registration((o['meshes'] if tn == 'ours' else n['meshes'])[k], k, 19, o['names'].index(k))
sw, dp = 'mini_sweeper', 'mini_dust_pan'
Ra = S[('new', sw)][0][:3, :3] @ S[('ours', sw)][0][:3, :3].T   # rotation taking ours anchor orientation to new at frame 0
off = S[('new', sw)][0][:3, 3] - Ra @ S[('ours', sw)][0][:3, 3]
raw = np.linalg.norm(S[('ours', dp)][:, :3, 3] - S[('new', dp)][:, :3, 3], axis=1) * 100
al = np.linalg.norm((S[('ours', dp)][:, :3, 3] @ Ra.T + off) - S[('new', dp)][:, :3, 3], axis=1) * 100
rot = [np.degrees(np.linalg.norm(R.from_matrix((Ra @ S[('ours', dp)][t][:3, :3]).T @ S[('new', dp)][t][:3, :3]).as_rotvec())) for t in range(len(al))]
sep = np.linalg.norm(S[('ours', dp)][0][:3, 3] - S[('ours', sw)][0][:3, 3]) * 100
res = dict(anchor_rotation_deg=float(np.degrees(np.linalg.norm(R.from_matrix(Ra).as_rotvec()))), sweeper_dustpan_separation_cm=float(sep),
           dustpan_pos_diff_cm_same_camera=dict(mean=float(raw.mean()), max=float(raw.max())), dustpan_pos_diff_cm_after_anchor_alignment=dict(mean=float(al.mean()), max=float(al.max())),
           dustpan_rot_diff_deg_after_anchor_alignment_mean=float(np.mean(rot)))
print(json.dumps(res, indent=1)); json.dump(res, open(sys.argv[1], 'w'), indent=1)
