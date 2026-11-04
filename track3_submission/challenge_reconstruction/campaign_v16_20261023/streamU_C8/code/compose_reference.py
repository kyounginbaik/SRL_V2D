"""Stream U (phase 6, ep8): convert the coworker's compose_eval tracks into OUR mesh body frames and OUR source world.
1. body conversion = the kit scorer's own registration (_register_object_body, seed 1000003*ep+101*slot) of our packed (budget)
   mesh onto their mesh, inverted, so that scoring the converted track against compose gives ~0 by construction;
2. world: one rigid transform W (compose world = rectified-left cam at frame 0 -> our final_recon source world) from the camera
   chain world_from_rgb[t] @ A_T_rect @ inv(cam0_T_cam[t]) (chordal mean over frames; per-frame spread reported);
3. diagnostics: per-frame shift vs our source poses; table contact in the streamB level setup frame.
usage: compose_reference.py OUT_DIR [--w-mode mean|frame0]"""
import sys, json, hashlib, argparse
from pathlib import Path
import numpy as np, pandas as pd, trimesh
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, '/data1/SRL_V2D/challenge_submissions_SRL/analysis_20261020/scripts')
import public_kit_score as K
CR = Path('/data1/SRL_V2D/challenge_reconstruction'); C16 = CR / 'campaign_v16_20261023'; EP = 8
SRC = CR / 'final_recon/episode_000008'; CMP = C16 / 'compose_target/episode_000008'
SUB = C16 / 'streamU_C8/scores/v21_baseline/sub.parquet'   # packed v21 submission: provides our exact packed (budget) mesh payloads
CAL = json.load(open('/data1/video_to_data_challenge/track_3/public/meta/camera_calibration.json'))['cameras']
T_AC = np.linalg.inv(np.array(CAL['ego_cam_a']['extrinsics_to_stereo_left'])) @ np.array(CAL['ego_cam_c']['extrinsics_to_stereo_left'])
CALIB = C16 / 'streamB_tilt38/ep8/calibrated_inputs/episodes/episode_000008/input_manifest.json'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--w-mode', default='mean'); a = ap.parse_args(); OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
M = K.load_metric('MP-SR')
src = np.load(SRC / 'object_poses.npz'); names = [str(n) for n in src['names']]; Wsrc = src['world_from_object'].astype(float); Wc = src['world_from_rgb'].astype(float)
Tn = len(Wsrc)
# --- body conversion (exact scorer registration) ---
sub = pd.read_parquet(SUB); sub = sub[sub.row_id.str.contains('/e000008/')]
conv = {}
for slot, n in enumerate(names):
    enc = sub[sub.row_id.str.endswith(f'/o{slot}')].object_mesh_b64.fillna('').astype(str); cand = M._decode_object_mesh(enc[enc.str.strip() != ''].iloc[0])
    m = trimesh.load(CMP / f'{n}_mesh_metric.obj', force='mesh'); m.merge_vertices()
    if len(m.faces) > 40000:
        try: m = m.simplify_quadric_decimation(face_count=40000)
        except Exception: pass
    m.remove_unreferenced_vertices(); v = np.asarray(m.vertices, np.float64); f = np.asarray(m.faces, np.int64)
    anchor, _ = M._surface_moments(v, f); ca, rot = M._register_object_body(cand, (v, f), anchor, 1_000_003 * EP + 101 * slot)
    # residual: our packed mesh mapped into their frame vs their surface
    from scipy.spatial import cKDTree
    cv = cand[0]; mapped = (cv - ca) @ rot.T + anchor   # p_their = R_rel (p_our - ca) + anchor  with R_rel = rot (body_rotation)
    d = cKDTree(trimesh.sample.sample_surface(m, 50000, seed=0)[0]).query(mapped)[0]
    ax, order = K.symmetry_axis(n, v)
    conv[n] = dict(candidate_anchor=ca, body_rotation=rot, their_anchor=anchor, residual_mm_median=float(np.median(d) * 1000), residual_mm_p90=float(np.percentile(d, 90) * 1000), symmetry_order=int(order), their_axis=ax)
# compose (their body) -> our body, in compose world
Tc = {n: np.load(CMP / f'{n}_trajectory.npz')['T_world_obj'].astype(float) for n in names}
ours_c = {}
for n in names:
    c = conv[n]; Rr = Tc[n][:, :3, :3]; Ra = Rr @ c['body_rotation']
    pa = Tc[n][:, :3, 3] + Rr @ c['their_anchor'] - Ra @ c['candidate_anchor']
    T = np.tile(np.eye(4), (Tn, 1, 1)); T[:, :3, :3] = Ra; T[:, :3, 3] = pa; ours_c[n] = T
# --- cup twist: the cup is scored rotationally symmetric, so its spin about the symmetry axis is free. Choose, per frame, the spin
# (about the axis line through the scorer's candidate anchor -> scored pose unchanged) closest to OUR v21 reference spin, smoothed,
# so the retargeted hand keeps its grasp instead of turning ~130 deg about the cup.
from scipy.ndimage import gaussian_filter1d
CUP_AXIS = K.symmetry_axis('beige_cup', np.asarray(trimesh.load(SRC / 'assets/beige_cup.obj', force='mesh').vertices))[0]
def twist_fix(Tc_our, Tref, axis, pivot):
    phi = np.zeros(len(Tc_our))
    for t in range(len(Tc_our)):
        Rrel = Tc_our[t, :3, :3].T @ Tref[t, :3, :3]   # body-frame rotation taking new -> ours
        rv = R.from_matrix(Rrel).as_rotvec(); q = R.from_matrix(Rrel).as_quat(); # swing-twist: twist angle about axis
        tw = 2 * np.arctan2(np.dot(q[:3], axis), q[3]); phi[t] = tw
    phi = gaussian_filter1d(np.unwrap(phi), 2.0, mode='nearest')
    out = Tc_our.copy()
    for t in range(len(out)):
        Rt = R.from_rotvec(axis * phi[t]).as_matrix(); G = np.eye(4); G[:3, :3] = Rt; G[:3, 3] = pivot - Rt @ pivot; out[t] = Tc_our[t] @ G
    return out, phi
# --- world transform via the camera chain ---
r = np.load(CR / 'eval_stereo_current_20261010/episode_000008/rectification.npz'); Mr = np.eye(4); Mr[:3, :3] = r['R1'].T; A_T_rect = T_AC @ Mr
C = np.load(CMP / 'camera_motion.npz')['cam0_T_cam'].astype(float)
Wt = np.einsum('tij,jk,tkl->til', Wc, A_T_rect, np.linalg.inv(C))
if a.w_mode == 'frame0': W = Wt[0].copy()
else:
    U, _, Vt = np.linalg.svd(Wt[:, :3, :3].sum(0)); Rm = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt; W = np.eye(4); W[:3, :3] = Rm; W[:3, 3] = Wt[:, :3, 3].mean(0)
spread_deg = np.degrees(R.from_matrix(np.einsum('ij,tkj->tik', W[:3, :3], Wt[:, :3, :3])).magnitude()); spread_cm = np.linalg.norm(Wt[:, :3, 3] - W[:3, 3], axis=1) * 100
new = np.stack([np.einsum('ij,tjk->tik', W, ours_c[n]) for n in names], 1)
k0 = names.index('beige_cup'); THEIR_AXIS = conv['beige_cup']['their_axis']   # from the scorer's (decimated) reference mesh
AX_OUR = conv['beige_cup']['body_rotation'].T @ THEIR_AXIS; AX_OUR /= np.linalg.norm(AX_OUR)   # registered image of the scorer's symmetry axis in our body
new[:, k0], PHI = twist_fix(new[:, k0], Wsrc[:, k0], AX_OUR, conv['beige_cup']['candidate_anchor'])
# --- diagnostics ---
diag = dict(W=W.tolist(), w_mode=a.w_mode, cup_twist_deg=dict(mean=float(np.degrees(PHI).mean()), min=float(np.degrees(PHI).min()), max=float(np.degrees(PHI).max())), W_spread_deg=dict(median=float(np.median(spread_deg)), max=float(spread_deg.max())), W_spread_cm=dict(median=float(np.median(spread_cm)), max=float(spread_cm.max())),
            body_conversion={n: dict(candidate_anchor=c['candidate_anchor'].tolist(), body_rotation=c['body_rotation'].tolist(), body_rotation_deg=float(np.degrees(R.from_matrix(c['body_rotation']).magnitude())),
                                     residual_mm_median=c['residual_mm_median'], residual_mm_p90=c['residual_mm_p90'], symmetry_order=c['symmetry_order']) for n, c in conv.items()}, objects={})
for k, n in enumerate(names):
    dp = np.linalg.norm(new[:, k, :3, 3] - Wsrc[:, k, :3, 3], axis=1) * 100
    dr = np.degrees(R.from_matrix(np.einsum('tji,tjk->tik', Wsrc[:, k, :3, :3], new[:, k, :3, :3])).magnitude())
    if n == 'beige_cup': diag.setdefault('cup_full_rotation_vs_v21_deg_after_twist', dict(mean=float(dr.mean()), max=float(dr.max())))
    if n == 'beige_cup':   # symmetric: axis angle only
        mm = trimesh.load(SRC / 'assets/beige_cup.obj', force='mesh'); ax, _ = K.symmetry_axis('beige_cup', np.asarray(mm.vertices))
        dr = np.degrees(np.arccos(np.clip(np.einsum('ti,ti->t', Wsrc[:, k, :3, :3] @ ax, new[:, k, :3, :3] @ ax), -1, 1)))
    diag['objects'][n] = dict(shift_cm_mean=float(dp.mean()), shift_cm_median=float(np.median(dp)), shift_cm_max=float(dp.max()), shift_cm_frame0=float(dp[0]), rot_deg_mean=float(dr.mean()), rot_deg_max=float(dr.max()), rot_deg_frame0=float(dr[0]),
                              shift_cm_per_frame=np.round(dp, 3).tolist(), rot_deg_per_frame=np.round(dr, 2).tolist())
# table contact in the streamB level setup frame
S = np.array(json.loads(CALIB.read_text())['setup_from_source'])
tab = {}
for k, n in enumerate(names):
    v = np.asarray(trimesh.load(SRC / f'assets/{n}.obj', force='mesh').vertices)
    for lab, P in (('ours', Wsrc), ('compose', new)):
        Ts = np.einsum('ij,tjk->tik', S, P[:, k]); z = (np.einsum('tij,vj->tvi', Ts[:, :3, :3], v) + Ts[:, None, :3, 3])[..., 2].min(1)
        tab.setdefault(n, {})[lab] = dict(bottom_z_cm_frame0=float(z[0] * 100), bottom_z_cm_last=float(z[-1] * 100), bottom_z_cm_min=float(z.min() * 100), bottom_z_cm_per_frame=np.round(z * 100, 2).tolist())
diag['table_level_frame'] = tab
np.savez(OUT / 'compose_in_source_world.npz', world_from_object=new, names=np.array(names), W=W, frames=np.arange(Tn))
diag['inputs'] = {str(p): sha(p) for p in [SRC / 'object_poses.npz', SUB, *[CMP / f'{n}_trajectory.npz' for n in names], *[CMP / f'{n}_mesh_metric.obj' for n in names], CMP / 'camera_motion.npz']}
json.dump(diag, open(OUT / 'diagnostics.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in diag.items() if k not in ('objects', 'table_level_frame', 'inputs')}, indent=1))
for n in names:
    o = diag['objects'][n]; print(n, {k: round(v, 2) for k, v in o.items() if not k.endswith('per_frame')})
    for lab in ('ours', 'compose'): t = tab[n][lab]; print('  table', lab, {k: round(v, 2) for k, v in t.items() if not k.endswith('per_frame')})
