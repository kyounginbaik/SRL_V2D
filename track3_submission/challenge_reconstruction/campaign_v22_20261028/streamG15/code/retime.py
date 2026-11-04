"""GT-free re-timing of a reconstruction export onto the stereo-camera (ego_cam_b/c) clock.
The RGB camera (ego_cam_a) lags ego_cam_b/c by delta frames (measured per episode from global image motion, camsync.py;
no GT). All per-frame poses (world_from_object, camera_from_object, world_from_rgb) are resampled at t + delta by SE(3)
interpolation (linear translation, slerp rotation), clamped at the clip ends. Masks/observed flags take the nearest frame.
Usage: retime.py SRC_DIR DST_DIR DELTA   (SRC_DIR holds object_poses.npz)"""
import sys, json, shutil, numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as Rot

def shift(T, d):
    F = len(T); d = np.broadcast_to(np.asarray(d, float), (F,)) if np.ndim(d) else d; t = np.clip(np.arange(F) + d, 0, F - 1); i0 = np.floor(t).astype(int); i1 = np.minimum(i0 + 1, F - 1); a = t - i0
    out = T.copy(); out[:, :3, 3] = (1 - a)[:, None] * T[i0, :3, 3] + a[:, None] * T[i1, :3, 3]
    R0 = Rot.from_matrix(T[i0, :3, :3]); R1 = Rot.from_matrix(T[i1, :3, :3]); rv = (R0.inv() * R1).as_rotvec()
    out[:, :3, :3] = (R0 * Rot.from_rotvec(rv * a[:, None])).as_matrix(); return out

def retime_npz(src, dst, d):
    z = dict(np.load(src, allow_pickle=False)); F = len(z['world_from_object']); n = z['world_from_object'].shape[1]
    d = np.asarray(d, float)[:F] if np.ndim(d) else d
    near = np.clip(np.round(np.arange(F) + d).astype(int), 0, F - 1)
    for k in ['world_from_object', 'camera_from_object']:
        if k in z: z[k] = np.stack([shift(z[k][:, i].astype(np.float64), d) for i in range(n)], 1).astype(z[k].dtype)
    if 'world_from_rgb' in z: z['world_from_rgb'] = shift(z['world_from_rgb'].astype(np.float64), d).astype(z['world_from_rgb'].dtype)
    for k in ['camera_valid', 'object_observed']:
        if k in z: z[k] = z[k][near]
    np.savez_compressed(dst, **z)

if __name__ == '__main__':
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    d = json.loads(Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG15/out/sync_delta_t.json').read_text())[sys.argv[3]] if '_' in sys.argv[3] else float(sys.argv[3])
    if dst.exists(): raise SystemExit(f'{dst} exists; never overwrite')
    dst.mkdir(parents=True)
    for f in src.iterdir():
        if f.name in ('object_poses.npz', 'public_score.json'): continue
        if f.is_file(): shutil.copy2(f, dst / f.name)
    retime_npz(src / 'object_poses.npz', dst / 'object_poses.npz', d)
    (dst / 'retime.json').write_text(json.dumps(dict(source=str(src), delta_frames=d if np.ndim(d)==0 else dict(key=sys.argv[3], values=sorted(set(d))), method='GT-free cam_a vs cam_b/c motion sync (G15 camsync.py)', ground_truth_used=False), indent=1))
    print('retimed', dst, d)
