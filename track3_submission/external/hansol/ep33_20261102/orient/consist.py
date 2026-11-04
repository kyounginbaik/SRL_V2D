import numpy as np, json, sys
sys.path.insert(0, '/home/hansol/v2d/outputs/spoon_stage_audit_20261029/code'); sys.path.insert(0, '/home/hansol/v2d/outputs/ep32_robot_20261031/code')
from spoon_hand_bowl import bowl_normal
from scipy.spatial.transform import Rotation as R
B = '/home/hansol/v2d/outputs/variants_20261030/recon_E14_best_B2A_S1_HGA/episode_000033'
z = np.load(B + '/object_poses.npz'); print('keys', z.files)
s = [str(n) for n in z['names']].index('wooden_spoon'); mj = json.load(open(B + '/meshes.json')); nb = bowl_normal(B + '/' + mj['wooden_spoon']['file'])[0]
Wo, Co, Wr = z['world_from_object'][:, s], z['camera_from_object'][:, s], z['world_from_rgb']
Ws = z['world_from_stored'] if 'world_from_stored' in z.files else None
for f in list(range(186, 204, 2)) + list(range(252, 268, 2)):
    A = Co[f]; Bm = np.linalg.inv(Wr[f]) @ Wo[f]
    fa = lambda T: float((T[:3, :3] @ nb) @ (-T[:3, 3] / np.linalg.norm(T[:3, 3])))
    dang = np.degrees(R.from_matrix(A[:3, :3].T @ Bm[:3, :3]).magnitude())
    extra = ''
    if Ws is not None:
        Cs = np.linalg.inv(Ws[f]) @ Wo[f]; extra = f' | via stored cam facing {fa(Cs):+.2f} (vs cam_from_obj {np.degrees(R.from_matrix(A[:3,:3].T @ Cs[:3,:3]).magnitude()):.1f} deg)'
    print(f'f{f}: facing cam_from_obj {fa(A):+.2f} | via world_from_rgb {fa(Bm):+.2f} | rot diff {dang:5.1f} deg{extra}')
