"""J8 ep19: quantify the dust-pan rest floor in world_st: target dust pan (our body frame, world_st) tilt / lowest z per frame vs the
physically settled dust pan, and sweeper-dust pan proximity at f0."""
import sys, numpy as np, mujoco, trimesh
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamF_margins/ep19/code')
from common19F import *
F = V; W = sys.argv[1] if len(sys.argv) > 1 else 'world_st'
z = np.load(F / W / 'episode_target/episode_000019/reference_trajectory.npz'); on = list(z['object_names'])
T = {n: to_T(z['object_positions'][:, on.index(n)], z['object_quaternions_wxyz'][:, on.index(n)]) for n in NAMES}
pk = packed_meshes(); Vb = {n: np.asarray(pk[n][0]) for n in NAMES}
st = np.load(F / 'dp_home_world_st.npz')['settled']; tw = np.load(F / 'twin_poses.npz')
ndp = st[:3, :3].T @ np.array([0, 0, -1.])   # dust pan body-frame down when resting
ap = trimesh.load('/dev/null') if False else None
lowz = {n: np.array([(Vb[n] @ T[n][f][:3, :3].T + T[n][f][:3, 3])[:, 2].min() for f in range(len(T[n]))]) for n in NAMES}
tilt = np.degrees(np.arccos(np.clip((np.einsum('tij,j->ti', T['mini_dust_pan'][:, :3, :3], ndp) @ [0, 0, -1.]), -1, 1)))
print('settled dp lowest z', (Vb['mini_dust_pan'] @ st[:3, :3].T + st[:3, 3])[:, 2].min() * 100)
for f in range(0, len(tilt), 10): print(f, 'dp lowz %.2f cm tilt %.1f | sw lowz %.2f' % (lowz['mini_dust_pan'][f] * 100, tilt[f], lowz['mini_sweeper'][f] * 100))
P0 = {n: Vb[n] @ T[n][0][:3, :3].T + T[n][0][:3, 3] for n in NAMES}
print('f0 sweeper-dp min dist cm', cKDTree(P0['mini_dust_pan']).query(P0['mini_sweeper'])[0].min() * 100)
# twin poses
for k in tw.files: print(k, tw[k][:3, 3].round(3), R.from_matrix(tw[k][:3, :3]).as_rotvec().round(3))
print('stable_twin vs target f0 sw', np.degrees(R.from_matrix(tw['stable_twin'][:3, :3].T @ T['mini_sweeper'][0][:3, :3]).magnitude()), np.linalg.norm(tw['stable_twin'][:3, 3] - T['mini_sweeper'][0][:3, 3]) * 100)
