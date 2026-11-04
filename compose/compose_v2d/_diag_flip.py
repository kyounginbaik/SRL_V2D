"""Does the 180-deg flipped hypothesis score better than the tracked pose under an UNTRIMMED residual? (ep12 blue_cup)"""
import numpy as np, sys, cv2, json
sys.path.insert(0, 'compose_v2d'); sys.path.insert(0, 'scripts')
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
from common import gt_world_T_obj, work_dir, load_meta, mono_masks_dir
from track_compose import refine_6dof, _safe_R, spill_fraction
from fit_mesh_scale import mesh_samples
ep, obj, tag = int(sys.argv[1]), sys.argv[2], sys.argv[3]
wd = work_dir(ep, obj); td = wd / f'track{tag}'; meta = load_meta(ep, obj); K = np.array(meta['K_rect'])
z = np.load(td / 'poses.npz'); E = z['E']; pcz = np.load(wd / 'pc_cache.npz')
g = mesh_samples(str(td / 'mesh_metric.obj'), 20000); tree = cKDTree(g)
c = g.mean(0); w, U = np.linalg.eigh(np.cov((g - c).T))
gaps = [abs(w[i] - w[(i + 1) % 3]) + abs(w[i] - w[(i + 2) % 3]) - abs(w[(i + 1) % 3] - w[(i + 2) % 3]) for i in range(3)]
ax = U[:, int(np.argmax(gaps))]; perp = U[:, (int(np.argmax(gaps)) + 1) % 3]
Flip = Rot.from_rotvec(np.pi * perp).as_matrix()          # 180 deg about an axis perpendicular to the revolution axis (rim <-> bottom)
gt, _ = gt_world_T_obj(ep, obj); T = np.einsum('nij,jk->nik', z['icp_track_6dof'], z['T_ours_from_scan'])
A = gt[0] @ np.linalg.inv(T[0]); ow = np.einsum('ij,njk->nik', A, T)
md = mono_masks_dir(ep, obj)
def scores(P, R, t, f):
    d, _ = tree.query((P - t) @ R, k=1)
    m = cv2.imread(str(md / f'obj_{f:05d}.png'), 0); mask_dil = cv2.dilate((m > 0).astype(np.uint8), np.ones((21, 21), np.uint8)) > 0
    Ei = np.linalg.inv(E[f]); Rc, tc = Ei[:3, :3] @ R, Ei[:3, :3] @ t + Ei[:3, 3]        # pose in the frame-f camera for the spill test
    return np.mean(np.sort(d)[:int(0.9 * len(d))]) * 1000, d.mean() * 1000, np.percentile(d, 95) * 1000, spill_fraction(g, Rc, tc, mask_dil, K) * 100
print('  f   tracked: trim90 / mean / p95 (mm)  spill%   |  flipped+ICP: trim90 / mean / p95  spill%   | ori err tracked vs GT (deg)')
for f in range(60, len(T), 10):
    P = pcz[f'pc_{f}'][:, :3].astype(np.float64) @ E[f, :3, :3].T + E[f, :3, 3]
    R0, t0 = z['icp_track_6dof'][f][:3, :3], z['icp_track_6dof'][f][:3, 3]
    s0 = scores(P, R0, t0, f)
    # flipped hypothesis: rotate 180 deg about perp axis through the mesh centre, then local ICP
    Rf = _safe_R(R0 @ Flip); tf = t0 + R0 @ c - Rf @ c
    Rf2, tf2, _, _ = refine_6dof(P, g, tree, Rf, tf, iters=40)
    s1 = scores(P, Rf2, tf2, f)
    e = np.degrees(np.linalg.norm(Rot.from_matrix(_safe_R(ow[f, :3, :3] @ gt[f, :3, :3].T)).as_rotvec()))
    print(f'{f:3d}   {s0[0]:5.2f} / {s0[1]:5.2f} / {s0[2]:5.2f}   {s0[3]:4.1f}    |   {s1[0]:5.2f} / {s1[1]:5.2f} / {s1[2]:5.2f}   {s1[3]:4.1f}    |   {e:5.1f}')

# symmetry-aware check: angle between the cup axes (ours vs GT, both in the scan frame)
import trimesh
from common import gt_mesh_path
gm = trimesh.load(gt_mesh_path(obj), force='mesh'); v = np.asarray(gm.vertices); cg = v.mean(0); wg, Ug = np.linalg.eigh(np.cov((v - cg).T))
gg = [abs(wg[i] - wg[(i + 1) % 3]) + abs(wg[i] - wg[(i + 2) % 3]) - abs(wg[(i + 1) % 3] - wg[(i + 2) % 3]) for i in range(3)]; axg = Ug[:, int(np.argmax(gg))]
print('\naxis (tilt) error of the tracked cup vs GT, symmetry-aware, every 10 frames:')
print(' '.join(f'{f}:{np.degrees(np.arccos(np.clip(abs((ow[f,:3,:3]@axg)@(gt[f,:3,:3]@axg)),-1,1))):.0f}' for f in range(0, len(T), 10)))
