"""Pitcher body-frame position bias learned on the two public pitcher clips (scorer-side error, our mesh),
expressed in OUR pitcher body frame via the kit registration rotation (rotation only, no scan geometry enters the mesh or poses).
LOO validation with bscore on pub21/held22, then writes the eval correction vector. usage: pitcher_bias.py OUT.json"""
import sys, json, numpy as np
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/public_gt_check_20261020/code/boards')
import bscore
from bscore import K
import trimesh
C = '/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/canonical'; O = '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG11_mp2/out'
M = bscore.mods()['MP-SR']
def L(p):
    m = trimesh.load(p, force='mesh'); m.merge_vertices(); return np.asarray(m.vertices, float), np.asarray(m.faces, np.int64)
from v2dlb.mesh_budget import budget_mesh
ours = f'{C}/public/episode_000021/objects/water_pitcher/mesh.obj'
mv, mf = budget_mesh(ours, 4096, 4096); cand = (np.asarray(mv, float), np.asarray(mf, np.int64))
g = K.reference_geometry(['water_pitcher'], bscore.mods()['AUC'])
Rb = {}
for ep, slot in [(21, 1), (22, 1)]:
    _, R = M._register_object_body(cand, (g['v0'], g['f0']), g['anchor0'], 1_000_003 * ep + 101 * slot); Rb[ep] = R
b = {ep: np.load(f'{O}/eb_{ep}_water_pitcher.npy')[0].mean(0) / 100 for ep in (21, 22)}
t = {ep: -Rb[ep].T @ b[ep] for ep in (21, 22)}
print('Rb diff deg', np.degrees(np.arccos(np.clip((np.trace(Rb[21].T @ Rb[22]) - 1) / 2, -1, 1))), 't21 cm', (t[21] * 100).round(2), 't22 cm', (t[22] * 100).round(2))
res = {}
for ep, root, other in [(21, f'{C}/public/episode_000021', 22), (22, f'{C}/public_heldout/episode_000022', 21)]:
    d = np.load(f'{root}/exports/r8_base/object_poses.npz'); names = [str(n) for n in d['names']]; mj = json.load(open(f'{root}/exports/r8_base/meshes.json'))
    for lab, tv in [('base', np.zeros(3)), ('loo', t[other]), ('both', (t[21] + t[22]) / 2)]:
        W = d['world_from_object'].copy(); s = names.index('water_pitcher'); T = np.eye(4); T[:3, 3] = tv; W[:, s] = W[:, s] @ T
        e, sc = bscore.score(ep, {n: (W[:, i], mj[n]) for i, n in enumerate(names)}, metrics=True)
        res[f'{ep}_{lab}'] = dict(err=e, MPPE=sc.get('MPPE'), AUC=sc.get('AUC')); print(ep, lab, e, 'MPPE %.3f AUC %.4f' % (sc['MPPE'], sc['AUC']), flush=True)
res['t_eval_cm'] = ((t[21] + t[22]) / 2 * 100).tolist()
json.dump(res, open(sys.argv[1], 'w'), indent=1)
