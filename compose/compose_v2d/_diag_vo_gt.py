"""GT-based ego-motion check (public only): for an object our camera-frame track is good for, the GT world pose plus our
camera-frame pose implies the camera trajectory; compare its relative motion cam0_T_cam(f) with the VO's."""
import sys, os, numpy as np
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import work_dir, gt_world_T_obj
ep, obj, tag, f0, f1 = int(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
z = np.load(work_dir(ep, obj) / f'track{tag}/poses.npz'); E = z['E']; Tos = z['T_ours_from_scan']
ours = np.einsum('nij,jk->nik', z['icp_track_6dof'], Tos)
r = gt_world_T_obj(ep, obj); gt = r[0] if isinstance(r, tuple) else r
n = min(len(E), len(gt), f1)
cam_T_scan = np.einsum('nij,njk->nik', np.linalg.inv(E[:n]), ours[:n])
W_T_cam = np.einsum('nij,njk->nik', gt[:n], np.linalg.inv(cam_T_scan))
imp = np.einsum('ij,njk->nik', np.linalg.inv(W_T_cam[f0]), W_T_cam)            # implied cam_f0_T_cam(f)
vo = np.einsum('ij,njk->nik', np.linalg.inv(E[f0]), E[:n])
D = np.einsum('nij,njk->nik', np.linalg.inv(vo), imp)
ang = np.degrees(np.linalg.norm(Rot.from_matrix(D[:, :3, :3]).as_rotvec(), axis=1)); tr = np.linalg.norm(D[:, :3, 3], axis=1) * 100
vo_ang = np.degrees(np.linalg.norm(Rot.from_matrix(vo[:, :3, :3]).as_rotvec(), axis=1)); gt_static = np.linalg.norm(gt[f0:n, :3, 3] - gt[f0, :3, 3], axis=1).max() * 100
sel = slice(f0, n)
print(f'ep{ep} {obj} {tag} frames {f0}-{n}: GT object moved max {gt_static:.1f} cm | VO rot max {vo_ang[sel].max():.1f} deg | implied-vs-VO diff: rot median {np.median(ang[sel]):.1f} / max {ang[sel].max():.1f} deg, trans median {np.median(tr[sel]):.1f} / max {tr[sel].max():.1f} cm')
for f in range(f0, n, max(1, (n - f0) // 6)): print(f'   f{f}: VO rot {vo_ang[f]:.1f} deg | diff rot {ang[f]:.1f} deg trans {tr[f]:.1f} cm')
