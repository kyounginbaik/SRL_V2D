"""K33-recon3 (GT-free hypothesis grid): ep33 truth candidates = our v14 track with a CONSTANT spoon body rotation E (about the mesh COM,
axis = PCA long/mid/short of our mesh, +-mag). Physically: a consistent bias of our spoon pose (or of our mesh body frame vs the real
spoon) over the whole episode - invisible frame-to-frame, but ep33 has the largest scorer lever of all spoon episodes (out/regsens).
Writes targets OUT/<axis><sign><mag>/episode_000033. usage: egrid.py OUT"""
import sys, shutil, numpy as np, trimesh
from pathlib import Path
from scipy.spatial.transform import Rotation as Rot
src = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/target_final_v14_g17/episode_000033'); O = Path(sys.argv[1])
m = trimesh.load(src / 'wooden_spoon_mesh_metric.obj', force='mesh'); com = np.asarray(m.center_mass)
v = np.asarray(m.vertices); c0 = v - v.mean(0); w, U = np.linalg.eigh(c0.T @ c0 / len(c0))
T = np.load(src / 'wooden_spoon_trajectory.npz', allow_pickle=False)['T_world_obj']
for ax, a in [('long', U[:, 2]), ('mid', U[:, 1]), ('short', U[:, 0])]:
    for mag in [-14, -10, -7, 7, 10, 14]:
        E = Rot.from_rotvec(a * np.radians(mag)).as_matrix(); B = np.eye(4); B[:3, :3] = E; B[:3, 3] = com - E @ com
        d = O / f'{ax}{mag:+d}' / 'episode_000033'; d.mkdir(parents=True)
        for f in src.iterdir():
            if f.name != 'wooden_spoon_trajectory.npz': shutil.copy2(f, d / f.name)
        np.savez_compressed(d / 'wooden_spoon_trajectory.npz', T_world_obj=T @ B)
print('ok')
