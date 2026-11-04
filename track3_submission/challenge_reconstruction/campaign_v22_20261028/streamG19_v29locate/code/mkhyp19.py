"""G19 task 3: hypothesis npz (camera_from_object in our canonical RGB camera, RGB clock) for ev.py (G13, GT-free image evidence):
v14 (selected20_final_v14_g17 shifted back by the G15 delta to the RGB clock, as G17 audit), v29 (achieved rollout poses: v29 results
are on the v14 (b/c) clock in v14's world -> camera via v14 world_from_rgb then shifted), fro, c21 (spoon only: team mesh).
usage: mkhyp19.py EP"""
import sys, os, json, numpy as np
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG13/code')
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/streamG17_place/code')
from tracks import *
import tracks as TR
from g17lib import shift, delta_arr
ep = int(sys.argv[1]); O = '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG19_v29locate/out/hyp'; os.makedirs(O, exist_ok=True)
B = f'{R}/reconstruction_recovery_20261026/selected20_final_v14_g17/episode_{ep:06d}'
pv = np.load(f'{B}/object_poses.npz'); names = [str(n) for n in pv['names']]; F = len(pv['world_from_rgb'])
mv = [f'{B}/' + json.load(open(f'{B}/meshes.json'))[n]['file'] for n in names]
d = delta_arr('evaluation', ep, F)
def save(tag, C, ms): np.savez(f'{O}/ep{ep}_{tag}.npz', camera_from_object=C, names=np.array(names), meshes=np.array(ms)); print('saved', tag, C.shape, flush=True)
Cb = pv['camera_from_object'].astype(float)
save('v14', np.stack([shift(Cb[:, i], -np.asarray(d)) for i in range(len(names))], 1), mv)
# v29 achieved: results world = target_final_v14_g17 world (the rollout's reference); camera = v14 target camera
TG = '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/target_final_v14_g17/episode_%06d' % ep
Tt = {n: np.load(f'{TG}/{n}_trajectory.npz')['T_world_obj'] for n in names}
ach = package('/data1/SRL_V2D/challenge_submissions_SRL/track3_v29_20261030/all_20_episodes', ep)
# camera_from_world per frame from the bundle: Cb = cam_from_obj, T_world_obj (target) -> cam_from_world = Cb @ inv(Tt) (slot 0)
CW = Cb[:, 0] @ np.linalg.inv(Tt[names[0]][:F])
k0 = next((k for k, n in enumerate(names) if 'cup' not in n), 0)  # frame-0 rigid map rollout world -> target world (non-cup object)
G = Tt[names[k0]][0] @ np.linalg.inv(ach[k0][1][0]); Fm = min(F, min(len(a[1]) for a in ach))
Ca = np.stack([CW[:Fm] @ G @ a[1][:Fm] for a in ach], 1)
Fa = Ca.shape[0]; save('v29', np.stack([shift(Ca[:, i], -np.asarray(d)[:Fa] if np.ndim(d) else -d) for i in range(len(names))], 1),
                      [f'/data1/SRL_V2D/challenge_submissions_SRL/track3_v29_20261030/all_20_episodes/meshes/episode_{ep:06d}/{n}.obj' for n in names])
FD = f'{FRO}/episode_{ep:06d}' if ep in (14, 15, 16, 20, 25, 28, 30) else f'{R}/final_recon/episode_{ep:06d}'
pf = np.load(f'{FD}/object_poses.npz', allow_pickle=True); nf = [str(n) for n in pf['names']]
Cf = np.linalg.inv(pf['world_from_rgb'].astype(float))[:, None] @ pf['world_from_object']
save('fro', np.stack([Cf[:, nf.index(n)] for n in names], 1), [f'{FD}/assets/{n}.obj' for n in names])
if names == ['wooden_spoon'] or 'wooden_spoon' in names:
    D = f'{TGT["c21"]}/episode_{ep:06d}'; cm = np.load(f'{D}/camera_motion.npz')
    Cc = np.stack([cm['T_a_from_rectL'] @ np.linalg.inv(cm['cam0_T_cam']) @ np.load(f'{D}/{n}_trajectory.npz')['T_world_obj'] for n in names], 1)
    # non-spoon objects keep v14 (c21 meshes of pots/cups are scan templates; never rendered)
    V = np.stack([shift(Cb[:, i], -np.asarray(d)) for i in range(len(names))], 1); Fc = min(len(Cc), F)
    i = names.index('wooden_spoon'); X = V[:Fc].copy(); X[:, i] = Cc[:Fc, i]
    save('c21spoon', X, [mv[k] if k != i else f'{D}/wooden_spoon_mesh_metric.obj' for k in range(len(names))])
