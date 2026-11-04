"""Build a new selected20 bundle from a base bundle: GT-free re-timing of every episode onto the stereo-camera clock
(per-episode delta from camsync.py) and optional mesh swaps (body-frame mesh refinement from meshfit.py).
usage: build_bundle.py BASE OUT [--meshes MESHMAP.json]
MESHMAP: {"<ep>": {"<object>": "/path/to/refined.obj"}}; refined meshes are expressed in the same body frame as the
poses (v' = A v + b already applied to the vertices), so poses are untouched by a mesh swap."""
import sys, json, shutil, argparse, hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from retime import retime_npz
ap = argparse.ArgumentParser(); ap.add_argument('base', type=Path); ap.add_argument('out', type=Path); ap.add_argument('--meshes', type=Path)
ap.add_argument('--no-retime', action='store_true'); ap.add_argument('--camdir', type=Path, help='G15 camera_sync outputs: CAMDIR/evaluation_<ep>/camera.npz')
a = ap.parse_args()
if a.out.exists(): raise SystemExit('exists; never overwrite')
D = json.loads((Path(__file__).parent.parent / 'out/sync_delta_t.json').read_text())
MM = json.loads(a.meshes.read_text()) if a.meshes else {}
shutil.copytree(a.base, a.out)
changes = dict(base=str(a.base), retime={}, mesh_swaps={}, ground_truth_used=False,
               method='G15: camera path re-estimated with clock-consistent stereo lifting (camera_sync.py, if camera given), then poses resampled at t+delta (ego_cam_a -> ego_cam_b/c clock, GT-free optical-flow sync); mesh swaps from GT-free silhouette mesh refinement')
for ep in sorted(a.out.glob('episode_*')):
    e = int(ep.name[-6:])
    if a.camdir:
        z = dict(np.load(ep / 'object_poses.npz')); cn = a.camdir / f'evaluation_{e}' / 'camera.npz'
        cam = np.load(cn)['world_from_camera'][:len(z['world_from_rgb'])]
        assert np.abs(np.einsum('tij,tnjk->tnik', z['world_from_rgb'], z['camera_from_object']) - z['world_from_object']).max() < 1e-9
        z['world_from_rgb'] = cam; z['world_from_object'] = np.einsum('tij,tnjk->tnik', cam, z['camera_from_object'])
        np.savez_compressed(ep / 'object_poses.npz', **z)
        if (ep / 'camera.npz').exists(): shutil.move(ep / 'camera.npz', ep / 'camera_pre_g15.npz')
        shutil.copy2(cn, ep / 'camera.npz'); shutil.copy2(cn.parent / 'camera_report.json', ep / 'camera_report_g15.json')
        changes.setdefault('camera', {})[e] = str(cn)
    if not a.no_retime:
        d = D[f'evaluation_{e}']; tmp = ep / 'object_poses.g15tmp.npz'
        retime_npz(ep / 'object_poses.npz', tmp, d); tmp.replace(ep / 'object_poses.npz'); changes['retime'][e] = sorted(set(d)) if np.ndim(d) else d
    if str(e) in MM:
        mj = json.loads((ep / 'meshes.json').read_text())
        for n, p in MM[str(e)].items():
            dst = ep / 'meshes' / f'{n}.obj'; old = hashlib.md5(dst.read_bytes()).hexdigest()[:8]
            shutil.copy2(p, dst); mj[n]['g15_refined_from'] = p
            changes['mesh_swaps'].setdefault(e, {})[n] = dict(old_md5=old, new_md5=hashlib.md5(dst.read_bytes()).hexdigest()[:8], source=p)
        (ep / 'meshes.json').write_text(json.dumps(mj, indent=2))
(a.out / 'G15_CHANGES.json').write_text(json.dumps(changes, indent=1))
print(json.dumps(changes, indent=1))
