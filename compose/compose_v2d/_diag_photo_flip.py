"""Feasibility of a photometric face test for the wooden boards: project the vertex-coloured team mesh under the tracked pose
and under its short-axis flip (the geometric twin) into the undistorted RGB, z-buffer, and compare the normalised
correlation of sample gray vs image gray.  usage: _diag_photo_flip.py ep obj tag frames..."""
import sys, os, numpy as np, cv2, trimesh
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, work_dir, load_meta
ep, obj, tag = int(sys.argv[1]), sys.argv[2], sys.argv[3]; frames = [int(x) for x in sys.argv[4:]]
wd = work_dir(ep, obj); z = np.load(wd / f'track{tag}/poses.npz'); E = z['E']; T = z['icp_track_6dof']; s0 = float(z['scale'])
meta = load_meta(ep, obj); Ka = np.array(meta['K_a_undist']); Ta = np.array(meta['T_a_from_rectL'])
mesh_path = sorted((ROOT / f'data/drive_results/V2D/extracted/public_meshes_20261008/selected_public/{obj}').glob('*.obj'))[0]
if obj == 'wooden_piece_2': mesh_path = ROOT / 'data/drive_results/V2D/extracted/public_meshes_20261008/selected_public/wooden_piece_1/wooden_piece_1.obj'
m = trimesh.load(str(mesh_path), force='mesh'); vc = np.asarray(m.visual.vertex_colors)[:, :3].astype(np.float32)
P, fid = trimesh.sample.sample_surface(m, 60000, seed=0); P = np.asarray(P) * s0; col = vc[m.faces[fid]].mean(1).mean(1)      # gray
c = P.mean(0); w, U = np.linalg.eigh(np.cov((P - c).T)); U = U[:, ::-1]                                                     # pc0 long, pc1 short, pc2 normal
def ncc(R, t, img, s=2):
    X = (P @ R.T + t); Xr = X @ np.linalg.inv(E[f])[:3, :3].T + np.linalg.inv(E[f])[:3, 3]; Xa = Xr @ Ta[:3, :3].T + Ta[:3, 3]
    zz = Xa[:, 2]; u = (Ka[0, 0] * Xa[:, 0] / zz + Ka[0, 2]); v = (Ka[1, 1] * Xa[:, 1] / zz + Ka[1, 2])
    H, W = img.shape; ok = (zz > 0.1) & (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
    cell = (v[ok] // s).astype(int) * (W // s + 1) + (u[ok] // s).astype(int); order = np.argsort(-zz[ok])
    zb = {}; 
    for i in order: zb[cell[i]] = i                                                                 # nearest sample per cell wins
    idx = np.array(list(zb.values())); sel = np.nonzero(ok)[0][idx]
    g_img = cv2.remap(img, u[sel].astype(np.float32), v[sel].astype(np.float32), cv2.INTER_LINEAR).ravel().astype(np.float32)
    g_mod = col[sel]
    a_, b_ = g_img - g_img.mean(), g_mod - g_mod.mean()
    return float((a_ * b_).sum() / (np.linalg.norm(a_) * np.linalg.norm(b_) + 1e-9)), len(sel)
for f in frames:
    img = cv2.GaussianBlur(cv2.imread(str(wd / 'rgb_undist' / f'{f:05d}.jpg'), 0), (5, 5), 0)
    R, t = T[f, :3, :3], T[f, :3, 3]; out = [('current', R, t)]
    for i, name in enumerate(('flip-long', 'flip-short', 'rot180-normal')):
        Rm = Rot.from_rotvec(np.pi * U[:, i]).as_matrix(); Rh = R @ Rm; th = t + R @ c - Rh @ c; out.append((name, Rh, th))
    print(f'frame {f}: ' + ' | '.join(f'{nm} NCC {ncc(Rh, th, img)[0]:+.3f} (n={ncc(Rh, th, img)[1]})' for nm, Rh, th in out))
