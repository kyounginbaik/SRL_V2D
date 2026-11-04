"""Write hypothesis npz files (camera_from_object in our canonical camera) for ep20: v10, fro (old target), fro_cup (v10 pitcher + fro cup), fro_pit."""
import numpy as np, json
from tracks import *
ep = 20; O = '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG13/out/hyp'; import os; os.makedirs(O, exist_ok=True)
B = f'{R}/reconstruction_recovery_20261026/selected20_final_v10_g12/episode_000020'
pv = np.load(f'{B}/object_poses.npz'); names = [str(n) for n in pv['names']]
mv = [f'{B}/' + json.load(open(f'{B}/meshes.json'))[n]['file'] for n in names]
Cv = pv['camera_from_object']
Cf = np.stack([np.load(f'{FRO}/episode_000020/{n}_trajectory.npz')['T_rgb_obj'] for n in names], 1)
mf = [f'{FRO}/episode_000020/assets/{n}.obj' for n in names]
def save(tag, C, ms): np.savez(f'{O}/ep20_{tag}.npz', camera_from_object=C, names=np.array(names), meshes=np.array(ms))
save('v10', Cv, mv); save('fro', Cf, mf)
X = Cv.copy(); X[:, 0] = Cf[:, 0]; save('frocup', X, [mf[0], mv[1]])
X = Cv.copy(); X[:, 1] = Cf[:, 1]; save('fropit', X, [mv[0], mf[1]])
print('ok', Cv.shape, Cf.shape)
