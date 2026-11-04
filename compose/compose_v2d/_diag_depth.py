import numpy as np, sys
sys.path.insert(0, 'compose_v2d')
from common import gt_world_T_obj
ep, obj, tag = int(sys.argv[1]), sys.argv[2], sys.argv[3]
z = np.load(f'outputs/compose_ep{ep}_{obj}/track{tag}/poses.npz'); pc = np.load(f'outputs/compose_ep{ep}_{obj}/pc_cache.npz')
gt, _ = gt_world_T_obj(ep, obj); T = np.einsum('nij,jk->nik', z['icp_track_6dof'], z['T_ours_from_scan'])
A = gt[0] @ np.linalg.inv(T[0]); ow = np.einsum('ij,njk->nik', A, T)
n = min(len(gt), len(T))
d_gt = np.linalg.norm(gt[:n, :3, 3] - gt[0, :3, 3], axis=1); d_ours = np.linalg.norm(ow[:n, :3, 3] - ow[0, :3, 3], axis=1)
E = z['E']
print(' f  npts  z_med z_p10 z_p90 (m, cam frame)  resid_it(mm)  GT|disp from f0| cm  ours|disp| cm   cam moved cm')
for f in range(0, n, 10):
    P = pc[f'pc_{f}'][:, :3].astype(np.float32) if f'pc_{f}' in pc.files else np.zeros((0, 3), np.float32)
    # pc cache is stored in the raw rect-left(f) camera frame
    zs = P[:, 2] if len(P) else np.array([np.nan])
    ri = float(z['resid_it'][f]) if 'resid_it' in z.files else float('nan')
    print(f'{f:3d} {len(P):5d}  {np.nanmedian(zs):5.2f} {np.nanpercentile(zs, 10):5.2f} {np.nanpercentile(zs, 90):5.2f}            {ri:6.2f}        {d_gt[f]*100:5.1f}            {d_ours[f]*100:5.1f}        {np.linalg.norm(E[f, :3, 3])*100:5.1f}')
