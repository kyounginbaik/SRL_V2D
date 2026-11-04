"""Stream R ep32: per-frame contradiction audit of the reference + independent stereo evidence.
For every STEP-th frame (source reconstruction world -> setup frame via the adapter manifest, table top = setup z 0):
 reference: cup axis tilt vs table normal, cup bottom height (57bb FR mesh), spoon min z, spoon bowl depth/radial in cup frame,
            spoon-into-cup-wall penetration (vertices inside the wall solid, radial model).
 stereo (eval_stereo_current_20261010 depth_rgb, undistorted 1014x760): table plane fitted per frame (points outside the
            object/hand masks, below the cup rim height), trimmed ICP of the FR cup mesh to cup-mask depth (hands+spoon masked),
            trimmed ICP of the spoon mesh to spoon-mask depth; residuals init vs fit; silhouette IoU of the reference-posed meshes.
usage: audit_ep32.py OUT_JSON [--poses OBJECT_POSES_NPZ] [--step 5]"""
import sys, json, argparse, numpy as np, cv2, trimesh
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--poses'); ap.add_argument('--step', type=int, default=5); a = ap.parse_args()
FR = Path('/data1/SRL_V2D/challenge_reconstruction/final_recon/episode_000032')
EPD = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/expanded_batch/repaired_export_inputs/episode_000032')
DEP = Path('/data1/SRL_V2D/challenge_reconstruction/eval_stereo_current_20261010/episode_000032/depth_rgb')
O = Path('/data1/SRL_V2D/challenge_submissions/track3/eval_hawor_latest_20261008/outputs/episode_000032')
K = json.load(open(O / 'blue_cup/anycalib/undistorted_intrinsics.json')); s = 1014 / K['width']; fx, fy, cx, cy = K['fx']*s, K['fy']*s, K['cx']*s, K['cy']*s
z = np.load(a.poses or FR / 'object_poses.npz'); names = list(z['names']); W = z['world_from_object']; wfc = z['world_from_rgb']
S = np.asarray(json.load(open(EPD / 'input_manifest.json'))['setup_from_source'])
cupm = trimesh.load(FR / 'assets/blue_cup.obj', force='mesh'); spm = trimesh.load(FR / 'assets/wooden_spoon.obj', force='mesh')
cv_ = np.asarray(cupm.vertices); sv = np.asarray(spm.vertices)
cup_s = cupm.sample(12000, return_index=False); sp_s = spm.sample(6000)
# cup axis local z; top = wider end
rad = np.linalg.norm(cv_[:, :2] - cv_[:, :2].mean(0), axis=1); top_sign = 1 if rad[cv_[:, 2] > np.percentile(cv_[:, 2], 90)].mean() > rad[cv_[:, 2] < np.percentile(cv_[:, 2], 10)].mean() else -1
axis_l = np.array([0, 0, top_sign], float); ctr_l = np.r_[cv_[:, :2].mean(0), 0]
hh = (cv_ - ctr_l) @ axis_l; rim_h = hh.max(); bot_h = hh.min()
# radial wall profile: outer radius per height bin
bins = np.linspace(bot_h, rim_h, 15); prof = np.array([np.percentile(rad[(hh >= bins[i]) & (hh < bins[i+1])], 95) if ((hh >= bins[i]) & (hh < bins[i+1])).sum() > 5 else np.nan for i in range(14)])
wall_t = 0.004
# spoon bowl = end of the long X axis with larger width
hi, lo = sv[:, 0] > np.percentile(sv[:, 0], 85), sv[:, 0] < np.percentile(sv[:, 0], 15)
bowl_mask = hi if np.ptp(sv[hi, 1]) > np.ptp(sv[lo, 1]) else lo; bowl_l = sv[bowl_mask].mean(0); bowl_pts = sv[bowl_mask]
def masks(f):
    m = {}
    for key, p in [('cup', O / 'blue_cup/masks/1'), ('spoon', O / 'wooden_spoon/masks/1'), ('h2', O / 'blue_cup/masks/2'), ('h3', O / 'blue_cup/masks/3')]:
        im = cv2.imread(str(p / f'{f:06d}.png'), 0); m[key] = cv2.resize(im, (1014, 760), interpolation=cv2.INTER_NEAREST) > 0 if im is not None else np.zeros((760, 1014), bool)
    return m
def backproject(dep, sel, f):
    v, u = np.nonzero(sel); Z = dep[v, u]; P = np.stack([(u-cx)*Z/fx, (v-cy)*Z/fy, Z], 1); return P @ wfc[f][:3, :3].T + wfc[f][:3, 3]
def to_setup(P): return P @ S[:3, :3].T + S[:3, 3]
def kabsch(A, B):
    ca, cb = A.mean(0), B.mean(0); H = (A-ca).T @ (B-cb); U, _, Vt = np.linalg.svd(H); d = np.sign(np.linalg.det(Vt.T @ U.T)); Rm = Vt.T @ np.diag([1, 1, d]) @ U.T; return Rm, cb - Rm @ ca
def icp(model, pts, T, iters=40, trim=80):
    T = T.copy()
    for _ in range(iters):
        posed = model @ T[:3, :3].T + T[:3, 3]; dist, idx = cKDTree(posed).query(pts); keep = dist < np.percentile(dist, trim)
        Rm, t = kabsch(posed[idx[keep]], pts[keep]); T[:3, :3] = Rm @ T[:3, :3]; T[:3, 3] = Rm @ T[:3, 3] + t
    return T
def resid(model, pts, T): return float(np.median(cKDTree(model @ T[:3, :3].T + T[:3, 3]).query(pts)[0]) * 100)
def project(Pw, f):
    Pc = (Pw - wfc[f][:3, 3]) @ wfc[f][:3, :3]; return np.stack([fx*Pc[:, 0]/Pc[:, 2]+cx, fy*Pc[:, 1]/Pc[:, 2]+cy], 1), Pc[:, 2]
def silhouette(mesh, T, f):
    uv, zc = project(np.asarray(mesh.vertices) @ T[:3, :3].T + T[:3, 3], f); img = np.zeros((760, 1014), np.uint8)
    tri = np.rint(uv[mesh.faces]).astype(np.int32); cv2.fillPoly(img, list(tri), 1); return img > 0
def iou(A, B): return float((A & B).sum() / max((A | B).sum(), 1))
rows = []
for f in range(0, len(W), a.step):
    Tc, Ts = W[f, names.index('blue_cup')], W[f, names.index('wooden_spoon')]
    Tc_s, Ts_s = S @ Tc, S @ Ts
    r = dict(frame=f, t=f/20.)
    ax = Tc_s[:3, :3] @ axis_l; r['ref_cup_tilt_setup_deg'] = float(np.degrees(np.arccos(np.clip(ax[2], -1, 1))))
    cw = cv_ @ Tc_s[:3, :3].T + Tc_s[:3, 3]; r['ref_cup_bottom_z_cm'] = float(cw[:, 2].min()*100)
    sw = sv @ Ts_s[:3, :3].T + Ts_s[:3, 3]; r['ref_spoon_min_z_cm'] = float(sw[:, 2].min()*100)
    loc = (sw - Tc_s[:3, 3]) @ Tc_s[:3, :3] - ctr_l; h = loc @ axis_l; rr = np.linalg.norm(loc - np.outer(h, axis_l), axis=1)
    bi = np.clip(np.digitize(h, bins) - 1, 0, 13); ro = np.where(np.isnan(prof[bi]), np.nanmax(prof), prof[bi])
    inwall = (h > bot_h) & (h < rim_h) & (rr < ro) & (rr > ro - wall_t); r['spoon_in_wall_pen_cm'] = float(np.max(np.where(inwall, np.minimum(ro - rr, rr - (ro - wall_t)), 0))*100)
    inbase = (h > bot_h) & (h < bot_h + 0.006) & (rr < ro); r['spoon_into_floor_cm'] = float(np.max(np.where(inbase, h[inbase].max() - bot_h if inbase.any() else 0, 0))*100) if inbase.any() else 0.
    b = (bowl_l @ Ts_s[:3, :3].T + Ts_s[:3, 3]); bl = (b - Tc_s[:3, 3]) @ Tc_s[:3, :3] - ctr_l; bh = bl @ axis_l; br = np.linalg.norm(bl - bh*axis_l)
    r['bowl_below_rim_cm'] = float((rim_h - bh)*100); r['bowl_radial_cm'] = float(br*100); r['bowl_outside_rim_cm'] = float(max(0., br - prof[-1])*100)
    d = cv2.imread(str(DEP / f'{f:06d}.png'), -1).astype(np.float64); dep = 65535.0/d - 1.0; dep[d >= 65535] = np.nan
    ok = np.isfinite(dep) & (dep > 0.1) & (dep < 1.5); m = masks(f); hands = cv2.dilate((m['h2'] | m['h3']).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    objs = cv2.dilate((m['cup'] | m['spoon']).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    # table plane: points outside all masks within 30 cm (xy) of the cup, below 3 cm in setup
    P = to_setup(backproject(dep, ok & ~hands & ~objs, f)); near = (np.linalg.norm(P[:, :2] - Tc_s[:2, 3], axis=1) < 0.30) & (abs(P[:, 2]) < 0.03)
    Q = P[near]
    if len(Q) > 500:
        for _ in range(3):
            c = Q.mean(0); _, _, Vt = np.linalg.svd(Q - c, full_matrices=False); n = Vt[2] * np.sign(Vt[2, 2]); res = (Q - c) @ n; Q = Q[abs(res) < max(0.004, 2.5*np.median(abs(res)))]
        r['table_normal_tilt_deg'] = float(np.degrees(np.arccos(n[2]))); r['table_z_at_cup_cm'] = float((c[2] - (n[0]*(Tc_s[0, 3]-c[0]) + n[1]*(Tc_s[1, 3]-c[1]))/n[2])*100)
        r['ref_cup_tilt_vs_table_deg'] = float(np.degrees(np.arccos(np.clip(ax @ n, -1, 1))))
    else: n = np.array([0, 0, 1.])
    # cup ICP
    cpts = backproject(dep, ok & m['cup'] & ~hands & ~(cv2.dilate(m['spoon'].astype(np.uint8), np.ones((7, 7), np.uint8)) > 0), f)
    if len(cpts) > 300:
        cpts = cpts[cKDTree(cup_s @ Tc[:3, :3].T + Tc[:3, 3]).query(cpts)[0] < 0.05]
    if len(cpts) > 300:
        Tf = icp(cup_s, cpts, Tc); Tf_s = S @ Tf; axf = Tf_s[:3, :3] @ axis_l
        r.update(cup_npts=int(len(cpts)), cup_res_ref_cm=resid(cup_s, cpts, Tc), cup_res_fit_cm=resid(cup_s, cpts, Tf),
                 cup_fit_minus_ref_cm=((Tf_s[:3, 3] - Tc_s[:3, 3])*100).tolist(), cup_fit_axis_vs_ref_deg=float(np.degrees(np.arccos(np.clip(axf @ ax, -1, 1)))),
                 fit_cup_tilt_setup_deg=float(np.degrees(np.arccos(np.clip(axf[2], -1, 1)))), fit_cup_tilt_vs_table_deg=float(np.degrees(np.arccos(np.clip(axf @ n, -1, 1)))),
                 fit_cup_bottom_z_cm=float((cv_ @ Tf_s[:3, :3].T + Tf_s[:3, 3])[:, 2].min()*100))
        # tilt-constrained check: cup upright (axis = table normal) at the fitted position, best yaw -> residual
    # spoon
    spts = backproject(dep, ok & m['spoon'] & ~hands, f)
    if len(spts) > 150:
        spts = spts[cKDTree(sp_s @ Ts[:3, :3].T + Ts[:3, 3]).query(spts)[0] < 0.05]
    if len(spts) > 150:
        Tf = icp(sp_s, spts, Ts); dTs = np.linalg.inv(Ts) @ Tf
        bf = (S @ Tf)[:3, :3] @ bowl_l + (S @ Tf)[:3, 3]; blf = (bf - Tc_s[:3, 3]) @ Tc_s[:3, :3] - ctr_l; bhf = blf @ axis_l
        r.update(spoon_npts=int(len(spts)), spoon_res_ref_cm=resid(sp_s, spts, Ts), spoon_res_fit_cm=resid(sp_s, spts, Tf), spoon_fit_shift_cm=float(np.linalg.norm(Tf[:3, 3]-Ts[:3, 3])*100),
                 spoon_fit_rot_deg=float(np.degrees(R.from_matrix(dTs[:3, :3]).magnitude())), spoon_fit_roll_deg=float(np.degrees(R.from_matrix(dTs[:3, :3]).as_rotvec()[0])),
                 fit_bowl_below_rim_cm=float((rim_h - bhf)*100), fit_bowl_radial_cm=float(np.linalg.norm(blf - bhf*axis_l)*100))
    r['cup_iou_ref'] = iou(silhouette(cupm, Tc, f), m['cup'] | (m['cup'] & False)); r['spoon_iou_ref'] = iou(silhouette(spm, Ts, f), m['spoon'])
    rows.append(r); print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if not isinstance(v, list)}), flush=True)
json.dump(dict(poses=str(a.poses or FR / 'object_poses.npz'), rim_radius_cm=float(prof[-1]*100), cup_height_cm=float((rim_h-bot_h)*100), rows=rows), open(a.out, 'w'), indent=1)
