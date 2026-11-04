"""E31: per-frame spoon change target_E14_X vs target_final_v14_g17 (relative rotation angle, and its body-frame axis vs the mesh long axis)."""
import sys, numpy as np, trimesh
from scipy.spatial.transform import Rotation as R
C = '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028'
for V in sys.argv[1].split(','):
  for ep in map(int, sys.argv[2].split(',')):
    a = np.load(f'{C}/target_final_v14_g17/episode_{ep:06d}/wooden_spoon_trajectory.npz')['T_world_obj']
    b = np.load(f'{C}/target_E14_{V}/episode_{ep:06d}/wooden_spoon_trajectory.npz')['T_world_obj']
    m = trimesh.load(f'{C}/target_final_v14_g17/episode_{ep:06d}/wooden_spoon_mesh_metric.obj', force='mesh'); v = m.vertices - m.vertices.mean(0)
    ax = np.linalg.svd(v, full_matrices=False)[2][0]
    rel = R.from_matrix(np.einsum('nji,njk->nik', a[:, :3, :3], b[:, :3, :3])); rv = rel.as_rotvec(); ang = np.degrees(np.linalg.norm(rv, axis=1))
    al = np.abs(rv @ ax) / np.maximum(np.linalg.norm(rv, axis=1), 1e-9)
    print(V, ep, 'N', len(a), 'mean', ang.mean().round(1), 'frames>60', int((ang > 60).sum()), 'frames>120', int((ang > 120).sum()), 'axis|long|(>20deg frames)', al[ang > 20].mean().round(2) if (ang > 20).any() else None)
    print('   per 10f:', ' '.join(f'{ang[s:s+10].mean():.0f}' for s in range(0, len(a), 10)))
