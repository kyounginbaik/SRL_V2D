"""TSHIFT: time-shifted copy of an existing K33 bundle: for BOTH objects, world_from_object[t] := src[min(t+K, N-1)] and
camera_from_object likewise (camera path / world_from_rgb / frame_indices unchanged). GT-free. Motivation: official MPPE diagnostics
(ep33 rows re-indexed t->t+1 improved MPPE 1.561 -> 1.529) suggest the hidden reference runs ~1 frame ahead of our track.
usage: build_h33ts.py LABEL SRC_BUNDLE_DIR K  ->  bundles/LABEL/episode_000033 + targets/LABEL/episode_000033 (both objects + meshes)"""
import sys, json, shutil, hashlib, numpy as np
from pathlib import Path
EP = 33
K33 = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK33_ep33')
lab, SRC, K = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]); NM = f'episode_{EP:06d}'
out = K33 / 'bundles' / lab / NM; assert not out.exists(); shutil.copytree(SRC, out)
z = np.load(SRC / 'object_poses.npz', allow_pickle=False); d = {k: z[k] for k in z.files}
N = len(d['world_from_object']); idx = np.minimum(np.arange(N) + K, N - 1)
for key in ('world_from_object', 'camera_from_object', 'object_observed'):
    if key in d: d[key] = d[key][idx]
(out / 'object_poses.npz').unlink(); np.savez_compressed(out / 'object_poses.npz', **d)
rec = dict(stream='K33', label=lab, source=str(SRC), ground_truth_used=False, rule='time shift: pose[t] := source pose[min(t+K, N-1)] for all objects; camera unchanged',
           shift_frames=K, frames=int(N), object_poses_sha256=hashlib.sha256((out / 'object_poses.npz').read_bytes()).hexdigest())
(out / 'K33TS_CHANGES.json').write_text(json.dumps(rec, indent=1))
tg = K33 / 'targets' / lab / NM; tg.mkdir(parents=True)
mj = json.loads((out / 'meshes.json').read_text())
for i, name in enumerate([str(n) for n in d['names']]):
    np.savez_compressed(tg / f'{name}_trajectory.npz', T_world_obj=d['world_from_object'][:, i])
    mp = mj[name]['file'] if isinstance(mj[name], dict) else mj[name]; mp = Path(mp) if str(mp).startswith('/') else out / mp
    shutil.copy2(mp, tg / f'{name}_mesh_metric.obj')
print(lab, 'K', K, 'frames', N, 'objects', [str(n) for n in d['names']])
