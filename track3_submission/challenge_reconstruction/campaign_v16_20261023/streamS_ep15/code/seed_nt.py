"""Seed reference for pickup searches in a world with a re-targeted object track: the given seed reference (robot columns) with every
free-object qpos column replaced by the world's tracking reference (the immutable object targets). usage: seed_nt.py SEED.npz WORLD OUT.npz"""
import sys, numpy as np
from pathlib import Path
seed, world, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
z = np.load(seed); t = np.load(world / 'episodes/episode_000015/tracking_reference.npz')
jn = [str(x) for x in z['joint_names']]; assert jn == [str(x) for x in t['joint_names']]
q = z['qpos'].copy(); cols = [i for i, n in enumerate(jn) if n.startswith('wooden_spoon') or n.startswith('plastic_dish_rack')]
assert len(cols) == 14, cols; q[:, cols] = t['qpos'][:, cols]
d = {k: z[k] for k in z.files}; d['qpos'] = q; d['source'] = np.array(f'{seed} with object columns from {world}/episodes/episode_000015/tracking_reference.npz')
if out.exists(): raise SystemExit('exists'); 
np.savez(out, **d); print(out, 'object cols', cols, 'max change m/quat', np.abs(q[:, cols] - z['qpos'][:, cols]).max())
