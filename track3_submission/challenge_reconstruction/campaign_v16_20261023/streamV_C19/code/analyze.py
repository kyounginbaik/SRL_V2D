import json, numpy as np, sys
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamV_C19/code')
from common19 import *
packed = packed_meshes(); reg = registration(packed); comp = compose_tracks()
print({n: dict(chamfer_mm=round(reg[n]['chamfer_mm'], 2), rot_deg=round(float(np.degrees(R.from_matrix(reg[n]['rot']).magnitude())), 2)) for n in NAMES})
# our source reference in setup world
man = json.loads((EPDIR / 'input_manifest.json').read_text()); S = np.array(man['setup_from_source'])
d = np.load(SRC / 'object_poses.npz'); W = d['world_from_object'].astype(float); names = [str(x) for x in d['names']]; assert names == NAMES
ours = {n: np.einsum('ij,tjk->tik', S, W[:, i]) for i, n in enumerate(NAMES)}
# tracking reference consistency
import mujoco
tr = np.load(EPDIR / 'tracking_reference.npz'); jn = list(tr['joint_names'])
for n in NAMES:
    print(n, 'tracking_reference vs S@source max pos diff', np.abs(tr['qpos'][:, jn.index(n + '_free'):jn.index(n + '_free') + 3] - ours[n][:, :3, 3]).max() if n + '_free' in jn else 'no joint name')
ch = json.loads(open('/data1/SRL_V2D/challenge_reconstruction/recon_v15_20261022/episode_000019/change.json').read()); G = np.array(ch['G'])
cbody = {n: comp[n] @ reg[n]['M'] for n in NAMES}
e = scorer_errors(ours, comp, reg); print('ideal: our source ref (no relabel) vs compose', {n: (round(e[n]['cm'], 2), round(e[n]['deg'], 1), round(e[n]['cm_max'], 1)) for n in NAMES})
o2 = dict(ours); o2['mini_sweeper'] = ours['mini_sweeper'] @ G
e = scorer_errors(o2, comp, reg); print('ideal: our source ref (relabel G) vs compose', {n: (round(e[n]['cm'], 2), round(e[n]['deg'], 1)) for n in NAMES})
e = scorer_errors(cbody, comp, reg); print('self check compose-in-our-body', {n: (round(e[n]['cm'], 4), round(e[n]['deg'], 3)) for n in NAMES})
t = np.load(C16 / 'common_v2_candidates/C_ep19_ilc/episode_000019/backend/evaluation/execution/trajectory.npz'); on = list(t['object_names'])
ach = {n: to_T(t['object_positions'][:, on.index(n)], t['object_quaternions_wxyz'][:, on.index(n)]) for n in NAMES}
print('achieved frames', len(ach[NAMES[0]]))
e = scorer_errors(ach, comp, reg); print('v21 rollout (no relabel) vs compose', {n: (round(e[n]['cm'], 2), round(e[n]['deg'], 1)) for n in NAMES})
e2 = scorer_errors(ach, {n: ours[n] @ np.linalg.inv(reg[n]['M']) for n in NAMES}, reg); print('v21 rollout vs our ref via same alignment', {n: (round(e2[n]['cm'], 2), round(e2[n]['deg'], 1)) for n in NAMES})
# per-second timeline of ideal and achieved vs compose
ei = scorer_errors(ours, comp, reg)
for s in range(0, 235, 20): print(s / 20, [(round(ei[n]['cm_t'][s], 1), round(ei[n]['deg_t'][s], 0), round(e[n]['cm_t'][min(s, len(e[n]['cm_t']) - 1)], 1), round(e[n]['deg_t'][min(s, len(e[n]['deg_t'])-1)], 0)) for n in NAMES])
