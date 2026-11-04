"""Write a hypothesis npz (camera_from_object, names, meshes) for G13 ev.py from a bundle episode.
usage: mkhyp.py BUNDLE EP OUT.npz"""
import sys, numpy as np
B, ep, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
E = f'{B}/episode_{ep:06d}'
d = np.load(f'{E}/object_poses.npz'); names = [str(n) for n in d['names']]
np.savez(out, camera_from_object=d['camera_from_object'], names=np.array(names), meshes=np.array([f'{E}/meshes/{n}.obj' for n in names]))
print(out, d['camera_from_object'].shape)
