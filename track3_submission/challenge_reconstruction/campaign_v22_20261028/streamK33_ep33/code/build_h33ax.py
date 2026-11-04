"""AX-sweep (copy of build_h33.py): frame-0 body-frame rotation of the ep33 spoon about its MID or SHORT PCA axis through the COM, applied
on top of an existing K33 bundle (e.g. K33_H5r16, which already carries the -16 deg long-axis roll). Same ramp as build_h33:
  corr(t) = c for t <= t_s, c*(1-smooth) for t_s<t<t_e, 0 after.  Cup untouched (GT-free).
Axes: eigh of the centred vertex covariance of the spoon mesh: long = U[:,2] (sign as build_h33: largest-|component| positive),
mid = U[:,1], short = U[:,0], each sign-fixed so its largest-magnitude component is positive.
After the rotation, if the lowest spoon vertex is below the stereo support plane (canonical support_plane.npz) on a corrected frame, the
spoon is lifted along the plane normal so the lowest vertex sits 2 mm above the plane (lift recorded per frame).
usage: build_h33ax.py LABEL SRC_BUNDLE_DIR AXIS(mid|short) DEG t_s t_e  ->  bundles/LABEL/episode_000033 + targets/LABEL/episode_000033"""
import sys, json, shutil, hashlib, numpy as np, trimesh
from pathlib import Path
from scipy.spatial.transform import Rotation as Rot
K = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK33_ep33')
CAN = Path('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/canonical/evaluation/episode_000033')
lab, SRC, axname, c, ts, te = sys.argv[1], Path(sys.argv[2]), sys.argv[3], float(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
out = K / 'bundles' / lab / 'episode_000033'; assert not out.exists(); shutil.copytree(SRC, out)
z = np.load(SRC / 'object_poses.npz', allow_pickle=False); d = {k: z[k] for k in z.files}; KS = list(d['names']).index('wooden_spoon')
m = trimesh.load(out / 'meshes' / 'wooden_spoon.obj', force='mesh'); v = np.asarray(m.vertices); com = np.asarray(m.center_mass)
c0 = v - v.mean(0); w, U = np.linalg.eigh(c0.T @ c0 / len(c0))
def fix(a): return -a if a[np.argmax(np.abs(a))] < 0 else a
long, mid, short = fix(U[:, 2]), fix(U[:, 1]), fix(U[:, 0])
axis = {'mid': mid, 'short': short}[axname]
G = np.load(CAN / 'support_plane.npz', allow_pickle=False)['gravity_from_canonical']   # gravity frame: +z up, plane at z=0 (as build_h33 rest_roll)
sm = lambda u: np.clip(u, 0, 1) ** 3 * (10 - 15 * np.clip(u, 0, 1) + 6 * np.clip(u, 0, 1) ** 2)
corr = np.array([c if t <= ts else (c * (1 - sm((t - ts) / (te - ts))) if t < te else 0.0) for t in range(len(d['world_from_object']))])
lifts = {}
for t in np.flatnonzero(corr != 0):
    E = Rot.from_rotvec(axis * np.radians(corr[t])).as_matrix(); B = np.eye(4); B[:3, :3] = E; B[:3, 3] = com - E @ com
    d['world_from_object'][t, KS] = d['world_from_object'][t, KS] @ B
    d['camera_from_object'][t, KS] = d['camera_from_object'][t, KS] @ B
    P = G @ d['world_from_object'][t, KS]; vg = v @ P[:3, :3].T + P[:3, 3]; low = float(vg[:, 2].min())
    if low < 0.002:
        lift = 0.002 - low; nrm_world = np.linalg.inv(G)[:3, :3] @ np.array([0, 0, 1.0]); nrm_world /= np.linalg.norm(nrm_world)
        d['world_from_object'][t, KS][:3, 3] += nrm_world * lift
        cw = d['camera_from_object'][t, KS] @ np.linalg.inv(d['world_from_object'][t, KS] @ np.linalg.inv(np.eye(4)))  # placeholder, recomputed below
        lifts[int(t)] = float(lift)
# keep camera_from_object consistent with world_from_object via the per-frame camera (world_from_rgb if present)
if 'world_from_rgb' in d:
    for t in lifts: d['camera_from_object'][t, KS] = np.linalg.inv(d['world_from_rgb'][t]) @ d['world_from_object'][t, KS]
else:
    for t in lifts:
        Cw = z['camera_from_object'][t, KS] @ np.linalg.inv(z['world_from_object'][t, KS]); d['camera_from_object'][t, KS] = Cw @ d['world_from_object'][t, KS]
(out / 'object_poses.npz').unlink(); np.savez_compressed(out / 'object_poses.npz', **d)
rec = dict(stream='K33', label=lab, source=str(SRC), ground_truth_used=False, rule=f'body-frame rotation about the spoon {axname} PCA axis through the COM, on top of the source bundle',
           axis=axname, c_deg=c, t_s=ts, t_e=te, long_axis_body=long.tolist(), mid_axis_body=mid.tolist(), short_axis_body=short.tolist(),
           sign_convention='largest-magnitude component of each axis made positive', com_body=com.tolist(), support_plane=str(CAN / 'support_plane.npz'),
           frames_changed=int((corr != 0).sum()), lifts_m=lifts, object_poses_sha256=hashlib.sha256((out / 'object_poses.npz').read_bytes()).hexdigest())
(out / 'K33_CHANGES.json').write_text(json.dumps(rec, indent=1))
tg = K / 'targets' / lab / 'episode_000033'; tg.mkdir(parents=True)
np.savez_compressed(tg / 'wooden_spoon_trajectory.npz', T_world_obj=d['world_from_object'][:, KS])
V14T = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/target_final_v14_g17/episode_000033')
for fn in ('blue_cup_trajectory.npz', 'blue_cup_mesh_metric.obj'): shutil.copy2(V14T / fn, tg / fn)
shutil.copy2(out / 'meshes' / 'wooden_spoon.obj', tg / 'wooden_spoon_mesh_metric.obj')
print(lab, axname, c, 'axis', np.round(axis, 3).tolist(), 'changed', rec['frames_changed'], 'lifts', {k: round(x * 1000, 1) for k, x in lifts.items()})
