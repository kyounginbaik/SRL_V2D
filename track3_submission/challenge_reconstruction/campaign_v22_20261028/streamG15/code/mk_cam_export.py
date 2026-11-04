"""Export TAG_OUT = TAG_IN camera-frame object poses composed with the G15 clock-consistent camera path (cam/<key>/camera.npz)."""
import sys, json, shutil, numpy as np, kitx
from pathlib import Path
tin, tout = sys.argv[1:3]
for e in kitx.TUNE + kitx.HELD:
    src = kitx.root(e) / 'exports' / tin; dst = kitx.root(e) / 'exports' / tout
    if dst.exists(): continue
    dst.mkdir()
    for f in src.iterdir():
        if f.is_file() and f.name not in ('public_score.json', 'object_poses.npz'): shutil.copy2(f, dst / f.name)
    z = dict(np.load(src / 'object_poses.npz')); cam = np.load(f'../cam/public_{e}/camera.npz')['world_from_camera'][:len(z['world_from_rgb'])]
    old = json.loads((kitx.root(e) / 'camera_report.json').read_text())['median_inliers']; new = json.loads(open(f'../cam/public_{e}/camera_report.json').read())['median_inliers']
    z['world_from_rgb'] = cam; z['world_from_object'] = np.einsum('tij,tnjk->tnik', cam, z['camera_from_object'])
    np.savez_compressed(dst / 'object_poses.npz', **z)
    print(e, 'median inliers', old, '->', new)
