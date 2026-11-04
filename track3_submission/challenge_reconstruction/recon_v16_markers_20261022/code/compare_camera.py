"""Public GT: same object-in-camera track (stage B with the metric mesh), world placement through the monocular SLAM camera
path vs the stereo-PnP camera path. GT used only for scoring. usage: compare_camera.py EP OBJ"""
import sys, os, json, glob
import numpy as np, pandas as pd
from scipy.spatial.transform import Rotation as Rot
P = '/data1/SRL_V2D/challenge_reconstruction/public_gt_check_20261020'
sys.path.insert(0, '/data1/SRL_V2D/challenge_submissions_SRL/analysis_20261020/scripts')
import public_kit_score as K
from v2dlb.mesh_budget import budget_mesh
ep, obj = int(sys.argv[1]), sys.argv[2]
z = np.load(f'{P}/outputs/episode_{ep:06d}/{obj}/result_slam_gravity_aligned/result.npz'); c2w = z['camera_to_world_transform']
files = sorted(glob.glob(f'{P}/stereo/episode_{ep:06d}/poses_{obj}_metric/*.json')); n = min(len(files), len(c2w))
Tco = np.tile(np.eye(4), (n, 1, 1))
for i, f in enumerate(files[:n]):
    j = json.load(open(f)); q = np.asarray(j['rotation'], float); Tco[i, :3, :3] = Rot.from_quat(q[[1, 2, 3, 0]]).as_matrix(); Tco[i, :3, 3] = j['translation']
pnp = np.load(f'/data1/SRL_V2D/challenge_reconstruction/recon_v16_markers_20261022/egomotion/public_ep{ep}.npz'); Wp = pnp['world_from_rgb']
auc = K.load_metric('AUC'); mods = {nm: K.load_metric(nm) for nm in ['AUC', 'SP-SR', 'MP-SR', 'MPPE']}
gt, vis = K.gt_world_T_obj(ep, obj); m = min(n, len(gt), len(Wp))
mv, mf = budget_mesh(f'{P}/meshes/{obj}.obj', 4096, 4096); payload = K.encode(vertices=np.asarray(mv, np.float64), faces=np.asarray(mf, np.float64))
geom = K.encode(**K.reference_geometry([obj], auc)); res = {}
# camera path difference itself (relative to frame 0)
rs = np.einsum('ij,njk->nik', np.linalg.inv(c2w[0]), c2w[:m]); rp = np.einsum('ij,njk->nik', np.linalg.inv(Wp[0]), Wp[:m])
res['camera_travel_cm'] = dict(slam=float(np.linalg.norm(rs[:, :3, 3], axis=1).max() * 100), pnp=float(np.linalg.norm(rp[:, :3, 3], axis=1).max() * 100),
                               max_diff_cm=float(np.linalg.norm(rs[:, :3, 3] - rp[:, :3, 3], axis=1).max() * 100),
                               max_diff_deg=float(np.degrees((Rot.from_matrix(rs[:, :3, :3]) * Rot.from_matrix(rp[:, :3, :3]).inv()).magnitude()).max()))
for name, Wc in (('slam_camera', c2w), ('pnp_camera', Wp)):
    Tw = np.einsum('nij,njk->nik', Wc[:m], Tco[:m]); g = K.pose_cols(gt[:m]); o = K.pose_cols(Tw)
    sol = pd.DataFrame([dict(row_id=f'track_3/e{ep:06d}/w000/f{f:06d}/o0', episode_index=ep, world_index=0, frame_index=f, object_slot=0, object_name=obj, **{'ref_' + k: float(v[f]) for k, v in g.items()}, reference_geometry_b64='') for f in range(m)])
    sub = pd.DataFrame([dict(row_id=f'track_3/e{ep:06d}/w000/f{f:06d}/o0', **{k: float(v[f]) for k, v in o.items()}, **K.PROV, object_mesh_b64=payload if f == 0 else '') for f in range(m)])
    sol.loc[0, 'reference_geometry_b64'] = geom
    sc = {nm: float(md.score(sol.copy(), sub.copy(), 'row_id')) for nm, md in mods.items()}; err = K.aligned_errors(mods['MP-SR'], sol, sub)[obj]
    res[name] = dict(pos_cm=err[0], orient_deg=err[1], **sc)
    print(f'ep{ep} {obj} {name:12s}: {err[0]:5.2f} cm / {err[1]:5.1f} deg | AUC {sc["AUC"]:.3f} MP {sc["MP-SR"]:.0f} MPPE {sc["MPPE"]:.2f}', flush=True)
print('camera path', {k: round(v, 2) for k, v in res['camera_travel_cm'].items()})
os.makedirs('/data1/SRL_V2D/challenge_reconstruction/recon_v16_markers_20261022/public', exist_ok=True)
json.dump(res, open(f'/data1/SRL_V2D/challenge_reconstruction/recon_v16_markers_20261022/public/camera_compare_ep{ep}_{obj}.json', 'w'), indent=1)
