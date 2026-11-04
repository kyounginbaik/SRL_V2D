"""Stream R ep32: resting-spoon pose (0 .. pickup) from stereo depth: per-frame trimmed ICP of the spoon mesh (spoon mask, hands
masked), multi-start (reference pose and reference rolled +-15 deg about the long axis), in the setup frame; compares with the
reference and with the physically settled pose of a rollout (trajectory npz, setup frame).
usage: rest_spoon_fit.py OUT_JSON TRAJ_NPZ [--last-frame 44]"""
import sys, json, argparse, numpy as np, cv2, trimesh
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('traj'); ap.add_argument('--last-frame', type=int, default=44); a = ap.parse_args()
FR = Path('/data1/SRL_V2D/challenge_reconstruction/final_recon/episode_000032')
EPD = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/expanded_batch/repaired_export_inputs/episode_000032')
DEP = Path('/data1/SRL_V2D/challenge_reconstruction/eval_stereo_current_20261010/episode_000032/depth_rgb')
O = Path('/data1/SRL_V2D/challenge_submissions/track3/eval_hawor_latest_20261008/outputs/episode_000032')
K = json.load(open(O / 'blue_cup/anycalib/undistorted_intrinsics.json')); s = 1014 / K['width']; fx, fy, cx, cy = K['fx']*s, K['fy']*s, K['cx']*s, K['cy']*s
z = np.load(FR / 'object_poses.npz'); names = list(z['names']); W = z['world_from_object'][:, names.index('wooden_spoon')]; wfc = z['world_from_rgb']
S = np.asarray(json.load(open(EPD / 'input_manifest.json'))['setup_from_source'])
spm = trimesh.load(FR / 'assets/wooden_spoon.obj', force='mesh'); sp_s = spm.sample(6000, seed=0) if 'seed' in trimesh.sample.sample_surface.__code__.co_varnames else spm.sample(6000)
sv = np.asarray(spm.vertices)
tr = np.load(a.traj); tn = [str(x) for x in tr['object_names']]; ti = tn.index('wooden_spoon')
def kabsch(A, B):
    ca, cb = A.mean(0), B.mean(0); H = (A-ca).T @ (B-cb); U, _, Vt = np.linalg.svd(H); d = np.sign(np.linalg.det(Vt.T @ U.T)); Rm = Vt.T @ np.diag([1, 1, d]) @ U.T; return Rm, cb - Rm @ ca
def icp(model, pts, T, iters=50, trim=80):
    T = T.copy()
    for _ in range(iters):
        posed = model @ T[:3, :3].T + T[:3, 3]; dist, idx = cKDTree(posed).query(pts); keep = dist < np.percentile(dist, trim)
        Rm, t = kabsch(posed[idx[keep]], pts[keep]); T[:3, :3] = Rm @ T[:3, :3]; T[:3, 3] = Rm @ T[:3, 3] + t
    d = cKDTree(model @ T[:3, :3].T + T[:3, 3]).query(pts)[0]; return T, float(np.mean(np.sort(d)[:int(len(d)*0.8)])*100)
def resid(T, pts): d = cKDTree(sp_s @ T[:3, :3].T + T[:3, 3]).query(pts)[0]; return float(np.mean(np.sort(d)[:int(len(d)*0.8)])*100)
rows = []; fits = []
for f in range(0, a.last_frame + 1):
    d = cv2.imread(str(DEP / f'{f:06d}.png'), -1).astype(np.float64); dep = 65535.0/d - 1.0; dep[d >= 65535] = np.nan
    ld = lambda p: cv2.resize(cv2.imread(str(p / f'{f:06d}.png'), 0), (1014, 760), interpolation=cv2.INTER_NEAREST) > 0
    m = ld(O / 'wooden_spoon/masks/1'); hands = cv2.dilate((ld(O / 'blue_cup/masks/2') | ld(O / 'blue_cup/masks/3')).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    m = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    ok = np.isfinite(dep) & (dep > 0.1) & (dep < 1.5) & m & ~hands; v, u = np.nonzero(ok); Zd = dep[v, u]
    P = np.stack([(u-cx)*Zd/fx, (v-cy)*Zd/fy, Zd], 1) @ wfc[f][:3, :3].T + wfc[f][:3, 3]
    T0 = W[f]; P = P[cKDTree(sp_s @ T0[:3, :3].T + T0[:3, 3]).query(P)[0] < 0.04]
    best = None
    for roll in (0, -15, 15):
        Ti = T0.copy(); Ti[:3, :3] = T0[:3, :3] @ R.from_euler('x', roll, degrees=True).as_matrix(); Tf, e = icp(sp_s, P, Ti)
        if best is None or e < best[1]: best = (Tf, e, roll)
    Tf = best[0]; Tfs = S @ Tf; Trs = S @ T0
    Ta = np.eye(4); Ta[:3, :3] = R.from_quat(tr['object_quaternions_wxyz'][f, ti][[1, 2, 3, 0]]).as_matrix(); Ta[:3, 3] = tr['object_positions'][f, ti]
    rot = lambda A, B: float(np.degrees(R.from_matrix(A[:3, :3].T @ B[:3, :3]).magnitude()))
    r = dict(frame=f, npts=int(len(P)), res_ref_cm=resid(T0, P), res_fit_cm=best[1], start_roll=best[2], fit_vs_ref_deg=rot(Trs, Tfs), fit_vs_ref_cm=float(np.linalg.norm(Tfs[:3, 3]-Trs[:3, 3])*100),
             sim_vs_ref_deg=rot(Trs, Ta), sim_vs_fit_deg=rot(Ta, Tfs), sim_vs_fit_cm=float(np.linalg.norm(Ta[:3, 3]-Tfs[:3, 3])*100), fit_minz_cm=float((sv @ Tfs[:3, :3].T + Tfs[:3, 3])[:, 2].min()*100),
             res_sim_cm=resid(np.linalg.inv(S) @ Ta, P), fit_setup=Tfs.tolist())
    rows.append(r); print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if k != 'fit_setup'}), flush=True)
F = [np.array(r['fit_setup']) for r in rows]; Rm = R.from_matrix(np.stack([x[:3, :3] for x in F])).mean(); pm = np.median([x[:3, 3] for x in F], axis=0)
spread = [float(np.degrees((R.from_matrix(x[:3, :3]) * Rm.inv()).magnitude())) for x in F]
json.dump(dict(rows=rows, mean_fit_setup_rotation_quat_xyzw=Rm.as_quat().tolist(), median_fit_setup_position=pm.tolist(), fit_rotation_spread_deg=spread), open(a.out, 'w'), indent=1)
print('fit spread about the mean (deg): median %.1f p90 %.1f' % (np.median(spread), np.percentile(spread, 90)))
