"""Hypothesis npz (camera_from_object in our canonical RGB camera) for an episode: v10, fro (old target: final_recon_ours for 14-30 set,
final_recon for the earlier 13), c21 (compose 10-21; only when its meshes are team meshes - never for scan-mesh objects).
usage: mkhyp.py EP [c21]"""
import sys, numpy as np, json, os
from tracks import *
ep = int(sys.argv[1]); O = '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG13/out/hyp'; os.makedirs(O, exist_ok=True)
B = f'{R}/reconstruction_recovery_20261026/selected20_final_v10_g12/episode_{ep:06d}'
pv = np.load(f'{B}/object_poses.npz'); names = [str(n) for n in pv['names']]
def save(tag, C, ms): np.savez(f'{O}/ep{ep}_{tag}.npz', camera_from_object=C, names=np.array(names), meshes=np.array(ms)); print('saved', tag, C.shape)
save('v10', pv['camera_from_object'], [f'{B}/' + json.load(open(f'{B}/meshes.json'))[n]['file'] for n in names])
FD = f'{FRO}/episode_{ep:06d}' if ep in (14, 15, 16, 20, 25, 28, 30) else f'{R}/final_recon/episode_{ep:06d}'
pf = np.load(f'{FD}/object_poses.npz', allow_pickle=True); nf = [str(n) for n in pf['names']]
Cf = np.linalg.inv(pf['world_from_rgb'].astype(float))[:, None] @ pf['world_from_object']
save('fro', np.stack([Cf[:, nf.index(n)] for n in names], 1), [f'{FD}/assets/{n}.obj' for n in names])
if 'c21' in sys.argv:
    D = f'{TGT["c21"]}/episode_{ep:06d}'; cm = np.load(f'{D}/camera_motion.npz')
    Cc = np.stack([cm['T_a_from_rectL'] @ np.linalg.inv(cm['cam0_T_cam']) @ np.load(f'{D}/{n}_trajectory.npz')['T_world_obj'] for n in names], 1)
    F = min(len(Cc), len(pv['camera_from_object'])); save('c21', Cc[:F], [f'{D}/{n}_mesh_metric.obj' for n in names])
