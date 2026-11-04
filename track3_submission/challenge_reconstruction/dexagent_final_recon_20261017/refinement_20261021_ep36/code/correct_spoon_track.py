"""Episode 36: the spoon track takes the wrong roll branch between ~12.55 and ~14.65 s (mocap-ball reprojection: the 180-deg
roll twin about the spoon long axis matches the real balls there, the track matches before 12.4 s and after 14.8 s).
Correction: R_new = R * Rx(phi(t)), phi ramps 0->180 deg over [12.35, 12.60] s and 180->0 over [14.65, 14.90] s (smoothstep),
positions unchanged (the long axis passes through the body origin). Writes a corrected copy of the source episode dir.
usage: correct_spoon_track.py OUT_DIR"""
import sys, json, shutil, hashlib, numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
SRC = Path('/data1/SRL_V2D/challenge_reconstruction/final_recon/episode_000036'); OUT = Path(sys.argv[1])
A0, A1, B0, B1 = 12.35, 12.60, 14.65, 14.90
if OUT.exists(): raise SystemExit('exists ' + str(OUT))
shutil.copytree(SRC, OUT, symlinks=True)
z = dict(np.load(SRC / 'object_poses.npz', allow_pickle=False)); T = z['world_from_object'].copy(); n = len(T); t = np.arange(n) / 20.0
s = lambda u: np.clip(u, 0, 1) ** 3 * (10 - 15 * np.clip(u, 0, 1) + 6 * np.clip(u, 0, 1) ** 2)
phi = 180.0 * (s((t - A0) / (A1 - A0)) - s((t - B0) / (B1 - B0)))
for k in range(n):
    if phi[k] != 0: T[k, 0, :3, :3] = T[k, 0, :3, :3] @ R.from_euler('x', phi[k], degrees=True).as_matrix()
z['world_from_object'] = T; np.savez_compressed(OUT / 'object_poses.npz', **z)
if (OUT / 'wooden_spoon_trajectory.npz').exists():
    tr = dict(np.load(OUT / 'wooden_spoon_trajectory.npz', allow_pickle=False))
    for key in list(tr):
        if tr[key].shape[-2:] == (4, 4) and len(tr[key]) == n:
            base = tr[key].copy()
            if key.lower().startswith('t_rgb') or 'rgb' in key.lower(): tr[key] = np.linalg.inv(z['world_from_rgb']) @ T[:, 0]
            else: tr[key] = T[:, 0] if base.ndim == 3 else tr[key]
    np.savez_compressed(OUT / 'wooden_spoon_trajectory.npz', **tr)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
rec = dict(episode=36, object='wooden_spoon', method='180-deg roll about the local long X axis on the wrong-branch interval, smoothstep ramps',
           ramps_s=[[A0, A1], [B0, B1]], evidence='refinement_20261021_ep36/diag/marker_predict*.jpg (mocap balls), diag/depth_branch_track.json (weak depth)',
           original_object_poses_sha256=sha(SRC / 'object_poses.npz'), corrected_object_poses_sha256=sha(OUT / 'object_poses.npz'), frames_changed=int((phi != 0).sum()))
(OUT / 'spoon_track_correction_v9.json').write_text(json.dumps(rec, indent=2)); print(json.dumps(rec))
