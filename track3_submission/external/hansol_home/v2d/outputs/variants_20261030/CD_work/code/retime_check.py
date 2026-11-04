"""Check that a retimed bundle episode equals the base episode resampled at t + delta(t) (G15 piecewise clock), and report
the motion smoothness around clock steps plus the per-window difference profile. GT-free.
usage: retime_check.py EP BASE_BUNDLE RETIMED_BUNDLE SYNC_SUMMARY_JSON SYNCFINE_JSON"""
import sys, json, numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
ep, A, B, SS, SF = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
E = f'episode_{ep:06d}'
a = np.load(f'{A}/{E}/object_poses.npz'); b = np.load(f'{B}/{E}/object_poses.npz')
ca, cb = np.load(f'{A}/{E}/camera.npz'), np.load(f'{B}/{E}/camera.npz')
print('camera keys A', {k: ca[k].shape for k in ca.files}); print('camera keys B', {k: cb[k].shape for k in cb.files})
for k in a.files:
    if k in ('names', 'fps', 'frame_indices', 'camera_valid', 'object_observed'):
        print(k, 'equal' if np.array_equal(a[k], b[k]) else 'DIFF')
s = json.load(open(SS))[f'evaluation_{ep}']; print('sync profile', s)
F = len(a['camera_from_object']); dl = np.zeros(F)
for v, lo, hi in zip(s['values'], s['bounds'][:-1], s['bounds'][1:]): dl[lo:hi] = v
t = np.clip(np.arange(F) + dl, 0, F - 1)
names = [str(n) for n in a['names']]
for o, nm in enumerate(names):
    C = a['camera_from_object'][:, o]
    pos = np.array([np.interp(t, np.arange(F), C[:, i, 3]) for i in range(3)]).T
    rot = Slerp(np.arange(F), R.from_matrix(C[:, :3, :3]))(t)
    Cb = b['camera_from_object'][:, o]
    dp = np.linalg.norm(Cb[:, :3, 3] - pos, axis=1) * 100
    dr = (R.from_matrix(Cb[:, :3, :3]) * rot.inv()).magnitude() * 57.3
    print(f'{nm}: camera_from_object(B) vs A resampled at t+delta: max {dp.max():.3f} cm / {dr.max():.3f} deg (median {np.median(dp):.3f})')
    W = b['world_from_object'][:, o, :3, 3]; Wa = a['world_from_object'][:, o, :3, 3]
    va, vb = np.linalg.norm(np.diff(Wa, axis=0), axis=1) * 100, np.linalg.norm(np.diff(W, axis=0), axis=1) * 100
    acc_a, acc_b = np.linalg.norm(np.diff(Wa, 2, axis=0), axis=1) * 100, np.linalg.norm(np.diff(W, 2, axis=0), axis=1) * 100
    print(f'   world speed cm/frame p50/p95/max A {np.percentile(va, 50):.2f}/{np.percentile(va, 95):.2f}/{va.max():.2f}  B {np.percentile(vb, 50):.2f}/{np.percentile(vb, 95):.2f}/{vb.max():.2f}')
    print(f'   world accel cm/frame^2 p95/max A {np.percentile(acc_a, 95):.2f}/{acc_a.max():.2f}  B {np.percentile(acc_b, 95):.2f}/{acc_b.max():.2f}')
    if len(s['values']) > 1:
        k = s['bounds'][1]
        print(f'   speed around step f{k - 3}..f{k + 2}: A {np.round(va[k - 3:k + 3], 2)}  B {np.round(vb[k - 3:k + 3], 2)}')
    dw = np.linalg.norm(W - Wa, axis=1) * 100
    print('   |B-A| world cm per 20-frame window', np.round([dw[i:i + 20].mean() for i in range(0, F, 20)], 2))
cw = np.linalg.norm(cb['world_from_rgb'][:, :3, 3] - ca['world_from_rgb'][:, :3, 3], axis=1) * 100 if 'world_from_rgb' in ca.files else None
if cw is not None: print('camera path change cm mean/max', round(cw.mean(), 2), round(cw.max(), 2))
f = json.load(open(SF)); print('syncfine keys', list(f)[:12])
for k, v in f.items():
    print(' ', k, str(v)[:700])
