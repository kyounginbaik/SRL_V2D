"""Episode 33: which roll about the spoon long axis fits the stereo depth best? For each frame, masked stereo spoon points
(eval_stereo_current depth_rgb, 1014x760, 65535/(d+1); eval_hawor masks/1) are compared with the spoon mesh posed by the
reference track rolled by phi (body x). Reports the median point-to-surface distance per phi and the best phi."""
import json, sys, numpy as np, cv2, trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
S = '/data1/SRL_V2D/challenge_reconstruction/eval_stereo_current_20261010/episode_000033'; O = '/data1/SRL_V2D/challenge_submissions/track3/eval_hawor_latest_20261008/outputs/episode_000033/wooden_spoon'
FR = '/data1/SRL_V2D/challenge_reconstruction/final_recon/episode_000033'
K = json.load(open(O + '/anycalib/undistorted_intrinsics.json')); s = 1014 / K['width']; fx, fy, cx, cy = K['fx'] * s, K['fy'] * s, K['cx'] * s, K['cy'] * s
z = np.load(sys.argv[1] if len(sys.argv) > 1 else FR + '/object_poses.npz'); names = [str(x) for x in z['names']]; wfo = z['world_from_object'][:, names.index('wooden_spoon')]; wfc = z['world_from_rgb']
mv = trimesh.load(FR + '/assets/wooden_spoon_visual.obj', force='mesh').sample(15000)
phis = list(range(-90, 91, 15)) + [180]
print('  t   npts | best phi | median dist (mm) at phi=-45,-30,-15,0,15,30,45,180')
for f in range(48, 94, 2):
    d = cv2.imread(f'{S}/depth_rgb/{f:06d}.png', -1).astype(np.float64); dep = 65535.0 / d - 1.0; dep[d >= 65535] = np.nan
    m = cv2.imread(f'{O}/masks/1/{f:06d}.png', -1); m = cv2.resize(m, (1014, 760), interpolation=cv2.INTER_NEAREST) > 0
    m &= np.isfinite(dep) & (dep > .1) & (dep < 1.5)
    if m.sum() < 80: print(f'{f/20:4.1f} too few stereo points'); continue
    v, u = np.nonzero(m); Z = dep[v, u]; P = np.stack([(u - cx) * Z / fx, (v - cy) * Z / fy, Z], 1); Pw = P @ wfc[f][:3, :3].T + wfc[f][:3, 3]
    res = {}
    for ph in phis:
        Rm = wfo[f][:3, :3] @ R.from_euler('x', ph, degrees=True).as_matrix(); pts = mv @ Rm.T + wfo[f][:3, 3]
        res[ph] = float(np.median(cKDTree(pts).query(Pw)[0]) * 1000)
    b = min(res, key=res.get)
    print(f'{f/20:4.1f} {int(m.sum()):6d} | {b:5d}    | ' + ' '.join(f'{res[p]:5.1f}' for p in (-45, -30, -15, 0, 15, 30, 45, 180)))
