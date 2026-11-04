"""ep33: spoon orientation / position difference between two compose targets per 20 frames (and the rotation axis of the difference
expressed in the spoon body frame: |x| ~ 1 means a flip about the long axis). usage: ep33_segdiff.py TARGET_A_EP TARGET_B_EP"""
import sys, numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as Rt
a, b = Path(sys.argv[1]), Path(sys.argv[2])
Ta = np.load(a / 'wooden_spoon_trajectory.npz')['T_world_obj']; Tb = np.load(b / 'wooden_spoon_trajectory.npz')['T_world_obj']; n = min(len(Ta), len(Tb))
Ca = np.load(a / 'blue_cup_trajectory.npz')['T_world_obj']; Cb = np.load(b / 'blue_cup_trajectory.npz')['T_world_obj']
d = Rt.from_matrix(Ta[:n, :3, :3]).inv() * Rt.from_matrix(Tb[:n, :3, :3]); ang = np.degrees(d.magnitude()); rv = d.as_rotvec(); ax = rv / np.maximum(np.linalg.norm(rv, axis=1, keepdims=True), 1e-9)
dp = 100 * np.linalg.norm(Ta[:n, :3, 3] - Tb[:n, :3, 3], axis=1); dc = 100 * np.linalg.norm(Ca[:n, :3, 3] - Cb[:n, :3, 3], axis=1)
dca = np.degrees((Rt.from_matrix(Ca[:n, :3, :3]).inv() * Rt.from_matrix(Cb[:n, :3, :3])).magnitude())
print(f'mean spoon diff {ang.mean():.1f} deg, {dp.mean():.2f} cm; cup {dca.mean():.1f} deg, {dc.mean():.2f} cm')
print(' frames    | spoon deg (mean/max) | axis in spoon frame |x| |y| |z| | spoon cm | cup cm / deg')
for s in range(0, n, 20):
    e = min(s + 20, n); A = np.abs(ax[s:e]).mean(0)
    print(f' f{s:3d}-{e:3d} | {ang[s:e].mean():6.1f} / {ang[s:e].max():6.1f}    | {A[0]:.2f} {A[1]:.2f} {A[2]:.2f}             | {dp[s:e].mean():5.2f}   | {dc[s:e].mean():4.2f} / {dca[s:e].mean():4.1f}')
