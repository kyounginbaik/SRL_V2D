"""Hypothesis npz for G13 ev.py: a bundle episode's camera_from_object resampled at t + s (constant s frames; lerp + slerp).
usage: mkshift.py BUNDLE EP S OUT.npz"""
import sys, numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
B, ep, s, out = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
E = f'{B}/episode_{ep:06d}'
d = np.load(f'{E}/object_poses.npz'); names = [str(n) for n in d['names']]; C = d['camera_from_object']; F = len(C)
t = np.clip(np.arange(F) + s, 0, F - 1); X = np.repeat(np.eye(4)[None, None], F, 0).repeat(len(names), 1)
for o in range(len(names)):
    X[:, o, :3, 3] = np.array([np.interp(t, np.arange(F), C[:, o, i, 3]) for i in range(3)]).T
    X[:, o, :3, :3] = Slerp(np.arange(F), R.from_matrix(C[:, o, :3, :3]))(t).as_matrix()
np.savez(out, camera_from_object=X, names=np.array(names), meshes=np.array([f'{E}/meshes/{n}.obj' for n in names]))
