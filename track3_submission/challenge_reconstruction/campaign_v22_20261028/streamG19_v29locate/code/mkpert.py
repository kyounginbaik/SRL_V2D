"""GT-free detectability of the kit-lever perturbations: v14 RGB-clock hypothesis with one object's track given the constant body rotation
(about its surface centroid, principal axis pcK) that sens19 found sufficient to fail v29 MP. usage: mkpert.py EP OBJ PC DEG"""
import sys, numpy as np, trimesh
from scipy.spatial.transform import Rotation as Rot
S = '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG19_v29locate'
ep, obj, pc, deg = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
H = np.load(f'{S}/out/hyp/ep{ep}_v14.npz'); names = [str(n) for n in H['names']]; ms = [str(m) for m in H['meshes']]; C = H['camera_from_object'].copy()
i = names.index(obj); m = trimesh.load(ms[i], force='mesh'); c = m.centroid; w, U = np.linalg.eigh(np.cov((np.asarray(m.vertices) - c).T))
R = Rot.from_rotvec(U[:, pc] * np.radians(deg)).as_matrix(); X = np.eye(4); X[:3, :3] = R; X[:3, 3] = c - R @ c
C[:, i] = C[:, i] @ X
np.savez(f'{S}/out/hyp/ep{ep}_pert_{obj}_pc{pc}_{deg:g}.npz', camera_from_object=C, names=H['names'], meshes=H['meshes'])
