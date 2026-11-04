"""R33: f0 state of the target worlds (old recon HGA7b vs latest r6): table plane, cup and spoon poses, lowest mesh point above the table."""
import json, numpy as np, trimesh
from pathlib import Path
from scipy.spatial.transform import Rotation as Rot
R = Path('/data1/hansol/retarget33')
for lab in ('R33_HGA7b', 'R33_HGA_r6'):
    W = R / 'worlds' / lab / 'ep33/episode_corrected/episode_000033'; T = R / 'targets' / lab / 'episode_000033'
    mg = json.loads((W / 'metric_geometry.json').read_text()); print(f'\n=== {lab}\nmetric_geometry keys:', list(mg)[:20])
    for k, v in mg.items():
        if any(s in k.lower() for s in ('table', 'plane', 'support', 'ground', 'floor')): print('  ', k, json.dumps(v)[:300])
    z = np.load(W / 'reference_trajectory.npz', allow_pickle=True); print('reference_trajectory keys:', {k: z[k].shape for k in z.files})
    for n in ('blue_cup', 'wooden_spoon'):
        Tw = np.load(T / f'{n}_trajectory.npz')['T_world_obj']; v = np.asarray(trimesh.load(T / f'{n}_mesh_metric.obj', force='mesh').vertices, float)
        for f in (0, 20, 40):
            X = v @ Tw[f, :3, :3].T + Tw[f, :3, 3]
            print(f'  {n} f{f}: pos {np.round(100*Tw[f,:3,3],1).tolist()} cm  euler_xyz {np.round(Rot.from_matrix(Tw[f,:3,:3]).as_euler("xyz",True),1).tolist()}  mesh z min/max {100*X[:,2].min():.1f}/{100*X[:,2].max():.1f} cm')
