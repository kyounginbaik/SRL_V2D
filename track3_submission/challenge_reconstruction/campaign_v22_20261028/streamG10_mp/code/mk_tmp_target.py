"""Scratch target (compose layout, one episode) = target_final_v6 with the spoon rolled by a balltrack roll track
(same smoothing as build_v8.py). For calibration only; not a deliverable.
usage: mk_tmp_target.py EP TAG OUTDIR"""
import sys, shutil, numpy as np
from pathlib import Path
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG10_mp/code')
from g10lib import C
from scipy.spatial.transform import Rotation as Rot
ep, tag, out = int(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
src = Path(f'{C}/target_final_v6/episode_{ep:06d}'); dst = out / f'episode_{ep:06d}'; shutil.copytree(src, dst)
roll = np.load(f'{C}/streamG10_mp/out/balltrack_ep{ep}_{tag}_roll.npy').astype(float)
R = np.load(f'{C}/streamG10_mp/out/rollscan_ep{ep}.npz'); c, ax = R['centroid'], R['axis']
un = np.degrees(np.unwrap(np.radians(roll))); sm = np.array([un[max(0, t - 2):t + 3].mean() for t in range(len(un))]); sm[0] = 0
T = np.load(dst / 'wooden_spoon_trajectory.npz')['T_world_obj'].copy()
for t in range(len(T)):
    Rr = Rot.from_rotvec(np.radians(sm[t]) * ax).as_matrix(); X = np.eye(4); X[:3, :3] = Rr; X[:3, 3] = c - Rr @ c; T[t] = T[t] @ X
np.savez_compressed(dst / 'wooden_spoon_trajectory.npz', T_world_obj=T)
print('ok', dst)
