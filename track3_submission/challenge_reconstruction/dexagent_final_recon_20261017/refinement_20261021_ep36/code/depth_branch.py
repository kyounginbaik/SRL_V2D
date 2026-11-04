"""Per-frame stereo-depth test of the spoon roll branch: z-buffer the mesh surface (track pose vs 180-deg roll twin about
local X) into the 1014x760 depth frame and compare with measured stereo depth on the projected footprint.
usage: depth_branch.py OUT.json [--poses object_poses.npz] [--stride N]"""
import sys, json, numpy as np, cv2, trimesh
from scipy.spatial.transform import Rotation as R
a = sys.argv[1:]; poses = '/data1/SRL_V2D/challenge_reconstruction/final_recon/episode_000036/object_poses.npz'; stride = 2
if '--poses' in a: i = a.index('--poses'); poses = a[i + 1]; del a[i:i + 2]
if '--stride' in a: i = a.index('--stride'); stride = int(a[i + 1]); del a[i:i + 2]
out = a[0]; D = '/data1/SRL_V2D/challenge_reconstruction/final_recon/episode_000036'; S = '/data1/SRL_V2D/challenge_reconstruction/eval_stereo_current_20261010/episode_000036/depth_rgb/%06d.png'
z = np.load(poses); K = np.load(D + '/camera_motion.npz')['K']
m = trimesh.load(D + '/assets/wooden_spoon_visual.obj', force='mesh'); P = trimesh.sample.sample_surface(m, 60000, seed=0)[0]
twin = R.from_euler('x', 180, degrees=True).as_matrix(); H, Wd = 760, 1014
def zbuf(T):
    X = P @ T[:3, :3].T + T[:3, 3]; uv = X @ K.T; u = np.round(uv[:, 0] / uv[:, 2]).astype(int); v = np.round(uv[:, 1] / uv[:, 2]).astype(int)
    ok = (u >= 0) & (u < Wd) & (v >= 0) & (v < H) & (X[:, 2] > 0.05); zb = np.full((H, Wd), np.inf); np.minimum.at(zb, (v[ok], u[ok]), X[ok, 2]); return zb
rows = []
for k in range(0, len(z['world_from_object']), stride):
    enc = cv2.imread(S % k, cv2.IMREAD_UNCHANGED).astype(float); dm = np.where(enc >= 65535, np.nan, 65535.0 / np.maximum(enc, 1) - 1)
    Tc = np.linalg.inv(z['world_from_rgb'][k]) @ z['world_from_object'][k, 0]; Tt = Tc.copy(); Tt[:3, :3] = Tc[:3, :3] @ twin
    res = {}
    for tag, T in (('track', Tc), ('twin', Tt)):
        zb = zbuf(T); fp = np.isfinite(zb) & np.isfinite(dm); diff = dm[fp] - zb[fp]; near = np.abs(diff) < 0.03
        res[tag] = (float(np.median(np.abs(diff[near]))) if near.sum() > 30 else None, int(near.sum()))
    opening_cam = float((Tc[:3, :3] @ np.array([0, 0, -1.]))[2])   # <0: bowl opening faces the camera
    rows.append(dict(frame=k, t=k / 20, track_mad_mm=None if res['track'][0] is None else res['track'][0] * 1000, twin_mad_mm=None if res['twin'][0] is None else res['twin'][0] * 1000, n=res['track'][1], opening_z_cam=opening_cam))
json.dump(rows, open(out, 'w'), indent=1)
for r in rows[::max(1, 10 // stride)]:
    w = '-' if r['track_mad_mm'] is None or r['twin_mad_mm'] is None else ('TRACK' if r['track_mad_mm'] < r['twin_mad_mm'] else 'TWIN')
    print(f"t={r['t']:5.1f} track {r['track_mad_mm'] or float('nan'):5.1f} mm  twin {r['twin_mad_mm'] or float('nan'):5.1f} mm  n={r['n']:5d}  -> {w}")
