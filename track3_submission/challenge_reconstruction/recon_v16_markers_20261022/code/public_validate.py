"""Validate the marker-ball refinement on a public episode against mocap GT (GT used only for scoring).
usage: public_validate.py EP OBJ [init=metric|scaled] [learn_frames]"""
import sys, os, json, glob
import numpy as np, cv2, trimesh, pandas as pd
from scipy.spatial.transform import Rotation as Rot
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import marker_track as M
P = '/data1/SRL_V2D/challenge_reconstruction/public_gt_check_20261020'
sys.path.insert(0, P + '/code'); sys.path.insert(0, '/data1/SRL_V2D/challenge_submissions_SRL/analysis_20261020/scripts')
import public_kit_score as K
from v2dlb.mesh_budget import budget_mesh
ep, obj = int(sys.argv[1]), sys.argv[2]; init = sys.argv[3] if len(sys.argv) > 3 else 'metric'; nlearn = int(sys.argv[4]) if len(sys.argv) > 4 else 40
OUT = f'/data1/SRL_V2D/challenge_reconstruction/recon_v16_markers_20261022/public/episode_{ep:06d}_{obj}'; os.makedirs(OUT, exist_ok=True)
slot0 = json.load(open(P + '/episodes.json')); e = [x for x in slot0 if x['episode_id'] == ep][0]
stageA = f'{P}/outputs/episode_{ep:06d}/{e["objects"][0]}'          # camera / video live in the slot-0 run
z = np.load(f'{P}/outputs/episode_{ep:06d}/{obj}/result_slam_gravity_aligned/result.npz'); c2w = z['camera_to_world_transform']
Kj = json.load(open(stageA + '/anycalib/undistorted_intrinsics.json')); Kf = np.array([[Kj['fx'], 0, Kj['cx']], [0, Kj['fy'], Kj['cy']], [0, 0, 1.]])
tag = 'poses_' + obj + ('_metric' if init == 'metric' else '')
files = sorted(glob.glob(f'{P}/stereo/episode_{ep:06d}/{tag}/*.json')); n = min(len(files), len(c2w))
Tco = np.tile(np.eye(4), (n, 1, 1))
for i, f in enumerate(files[:n]):
    j = json.load(open(f)); q = np.asarray(j['rotation'], float); Tco[i, :3, :3] = Rot.from_quat(q[[1, 2, 3, 0]]).as_matrix(); Tco[i, :3, 3] = j['translation']
mesh_path = f'{P}/meshes/{obj}.obj' if init == 'metric' else f'{P}/outputs/episode_{ep:06d}/{obj}/provided_mesh_scaled.obj'
V = np.asarray(trimesh.load(mesh_path, force='mesh').vertices)
dep_dir = f'{P}/stereo/episode_{ep:06d}/depth_rgb'
def depth_fn(k, u, v):
    d = cv2.imread(f'{dep_dir}/{k:06d}.png', -1)
    if d is None: return None
    x, y = int(round(u / 2)), int(round(v / 2)); w = d[max(0, y - 1):y + 2, max(0, x - 1):x + 2].astype(float); w = w[w < 65535]
    if not len(w): return None
    zz = 65535.0 / w - 1; return float(np.min(zz))
reader = M.FrameReader(stageA + '/anycalib/undistorted.mp4')
balls, info = M.learn_layout(reader, Kf, Tco, V, list(range(min(nlearn, n))), depth_fn)
print('layout', len(balls), 'balls', info, flush=True)
T_tr, stats = M.track(reader, Kf, Tco, V, balls, list(range(n)), depth_fn=depth_fn)
T_sm = M.smooth_correction(T_tr, Tco, half=3)
acc = np.mean([s['accepted'] for s in stats]); print(f'accepted {acc:.2f} of {n} frames; median dets', np.median([s['n_det'] for s in stats]), flush=True)
np.savez(OUT + '/tracks.npz', T_init=Tco, T_marker=T_tr, T_smooth=T_sm, c2w=c2w[:n], balls=balls)
json.dump(dict(balls=balls.tolist(), info=info, stats=stats), open(OUT + '/stats.json', 'w'), indent=1)
# ---- score with the kit scorers (single object) ----
auc = K.load_metric('AUC'); mods = {nm: K.load_metric(nm) for nm in ['AUC', 'SP-SR', 'MP-SR', 'MPPE']}
gt, vis = K.gt_world_T_obj(ep, obj); m = min(n, len(gt))
mv, mf = budget_mesh(mesh_path if init == 'metric' else f'{P}/meshes/{obj}.obj', 4096, 4096); payload = K.encode(vertices=np.asarray(mv, np.float64), faces=np.asarray(mf, np.float64))
geom = K.encode(**K.reference_geometry([obj], auc)); res = {}
for name, Tc in (('init', Tco), ('marker', T_tr), ('marker_smooth', T_sm)):
    Tw = np.einsum('nij,njk->nik', c2w[:m], Tc[:m]); g = K.pose_cols(gt[:m]); o = K.pose_cols(Tw)
    sol = pd.DataFrame([dict(row_id=f'track_3/e{ep:06d}/w000/f{f:06d}/o0', episode_index=ep, world_index=0, frame_index=f, object_slot=0, object_name=obj, **{'ref_' + k: float(v[f]) for k, v in g.items()}, reference_geometry_b64='') for f in range(m)])
    sub = pd.DataFrame([dict(row_id=f'track_3/e{ep:06d}/w000/f{f:06d}/o0', **{k: float(v[f]) for k, v in o.items()}, **K.PROV, object_mesh_b64=payload if f == 0 else '') for f in range(m)])
    sol.loc[0, 'reference_geometry_b64'] = geom
    sc = {nm: float(md.score(sol.copy(), sub.copy(), 'row_id')) for nm, md in mods.items()}; err = K.aligned_errors(mods['MP-SR'], sol, sub)[obj]
    res[name] = dict(pos_cm=err[0], orient_deg=err[1], **sc); print(f'ep{ep} {obj} {name:13s}: {err[0]:5.2f} cm / {err[1]:5.1f} deg | AUC {sc["AUC"]:.3f} MP {sc["MP-SR"]:.0f} MPPE {sc["MPPE"]:.2f}', flush=True)
json.dump(res, open(OUT + '/scores.json', 'w'), indent=1)
