"""K33-recon3 scorer-side SENSITIVITY diagnostic (GT-free; no reference used): mimic a seed-dependent registration error E of the spoon
body rotation by right-multiplying our own v14 spoon target track by E about the mesh COM, then score the submitted rollout against it
with the kit-faithful scorer. Shows how many cm/deg a registration error of |E| costs in each spoon episode (ep33 vs passing ones).
usage: regsens.py OUTROOT  (writes targets OUTROOT/<lab>/episode_<ep>)"""
import sys, shutil, numpy as np, trimesh
from pathlib import Path
from scipy.spatial.transform import Rotation as Rot
V = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/target_final_v14_g17'); O = Path(sys.argv[1])
for ep in [33, 36, 5, 3, 6, 37]:
    src = V / f'episode_{ep:06d}'; m = trimesh.load(src / 'wooden_spoon_mesh_metric.obj', force='mesh'); com = np.asarray(m.center_mass)
    v = np.asarray(m.vertices); c0 = v - v.mean(0); w, U = np.linalg.eigh(c0.T @ c0 / len(c0))
    T = np.load(src / 'wooden_spoon_trajectory.npz', allow_pickle=False)['T_world_obj']
    for ax, a in [('long', U[:, 2]), ('mid', U[:, 1]), ('short', U[:, 0])]:
        for mag in [5, 10]:
            E = Rot.from_rotvec(a * np.radians(mag)).as_matrix(); B = np.eye(4); B[:3, :3] = E; B[:3, 3] = com - E @ com
            d = O / f'{ax}{mag}' / f'episode_{ep:06d}'; d.mkdir(parents=True, exist_ok=True)
            for f in src.iterdir():
                if f.name != 'wooden_spoon_trajectory.npz': shutil.copy2(f, d / f.name)
            np.savez_compressed(d / 'wooden_spoon_trajectory.npz', T_world_obj=T @ B)
print('ok')
