"""Shift a compose-layout ep33 target in time by K frames (both objects): T'[f] = T[clip(f - K)], meshes copied. K > 0 = target later.
usage: ep33_timeshift_target.py SRC_TARGET_EP_DIR OUT_TARGET_EP_DIR K"""
import sys, shutil, numpy as np
from pathlib import Path
src, out, k = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]); out.mkdir(parents=True, exist_ok=False)
for p in src.iterdir():
    if p.name.endswith('_trajectory.npz'):
        T = np.load(p)['T_world_obj']; idx = np.clip(np.arange(len(T)) - k, 0, len(T) - 1); np.savez_compressed(out / p.name, T_world_obj=T[idx])
    else: shutil.copy2(p, out / p.name)
print('shifted', k)
