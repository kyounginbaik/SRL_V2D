"""Public GT: the SAME camera-relative object tracks placed in the world through several camera paths, scored with the kit
scorers (GT only for scoring). Paths (all anchored at the stage-A frame-0 camera): slam, pnp, slamR_pnpT (SLAM rotation + PnP
translation), pnp_scaled (PnP translation scaled by the least-squares fit to SLAM), avg (mean translation, half-way rotation).
usage: compare_paths_public.py EP"""
import sys, os, json, glob, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy.spatial.transform import Rotation as Rot, Slerp
P = '/data1/SRL_V2D/challenge_reconstruction/public_gt_check_20261020'; O = '/data1/SRL_V2D/challenge_reconstruction/recon_v17_campath_20261022'
sys.path.insert(0, '/data1/SRL_V2D/challenge_submissions_SRL/analysis_20261020/scripts'); sys.path.insert(0, O + '/code')
import public_kit_score as K
from v2dlb.mesh_budget import budget_mesh
from static_cue import rest_mask, jitter
ep = int(sys.argv[1]); E = {x['episode']: x for x in json.load(open(P + '/episodes.json'))}[f'episode_{ep:06d}']; objs = E['objects']
c2w = np.load(f'{P}/outputs/episode_{ep:06d}/{objs[0]}/result_slam_gravity_aligned/result.npz')['camera_to_world_transform']
def cam_tracks():
    if ep in (23, 24):
        z = np.load(f'{P}/boards/episode_{ep:06d}/final_tracks.npz'); nm = [str(x) for x in z['names']]
        return {o: z['cam_from_object'][:, nm.index(o)] for o in objs}
    out = {}
    for o in objs:
        files = sorted(glob.glob(f'{P}/stereo/episode_{ep:06d}/poses_{o}_metric/*.json')); T = np.tile(np.eye(4), (len(files), 1, 1))
        for i, f in enumerate(files):
            j = json.load(open(f)); q = np.asarray(j['rotation'], float); T[i, :3, :3] = Rot.from_quat(q[[1, 2, 3, 0]]).as_matrix(); T[i, :3, 3] = j['translation']
        out[o] = T
    return out
Tco = cam_tracks(); Wp = np.load(f'{O}/public/egomotion/public_ep{ep}.npz')['world_from_rgb']
gts = {o: K.gt_world_T_obj(ep, o)[0] for o in objs}
m = min([len(c2w), len(Wp)] + [len(v) for v in Tco.values()] + [len(v) for v in gts.values()])
W0 = c2w[0]; rs = np.einsum('ij,njk->nik', np.linalg.inv(W0), c2w[:m]); rp = np.einsum('ij,njk->nik', np.linalg.inv(Wp[0]), Wp[:m])
ts, tp = rs[:, :3, 3], rp[:, :3, 3]; s = float((tp * ts).sum() / max((tp * tp).sum(), 1e-12))
def mk(Rr, t): T = np.tile(np.eye(4), (m, 1, 1)); T[:, :3, :3] = Rr; T[:, :3, 3] = t; return np.einsum('ij,njk->nik', W0, T)
key = np.arange(m); hal = np.array([Slerp([0, 1], Rot.from_matrix(np.stack([rs[k, :3, :3], rp[k, :3, :3]])))(0.5).as_matrix() for k in key])
paths = dict(slam=c2w[:m], pnp=mk(rp[:, :3, :3], tp), slamR_pnpT=mk(rs[:, :3, :3], tp), pnp_scaled=mk(rp[:, :3, :3], s * tp), avg=mk(hal, 0.5 * (ts + tp)))
auc = K.load_metric('AUC'); mods = {nm: K.load_metric(nm) for nm in ['AUC', 'SP-SR', 'MP-SR', 'MPPE']}
geom = K.encode(**K.reference_geometry(objs, auc)); res = dict(episode=ep, objects=objs, frames=m, pnp_scale_fit=s,
    camera=dict(slam_travel_cm=float(np.linalg.norm(ts, axis=1).max() * 100), pnp_travel_cm=float(np.linalg.norm(tp, axis=1).max() * 100),
                diff_mean_cm=float(np.linalg.norm(ts - tp, axis=1).mean() * 100), diff_max_cm=float(np.linalg.norm(ts - tp, axis=1).max() * 100)))
payload = {}
for o in objs:
    mv, mf = budget_mesh(f'{P}/meshes/{o}.obj', 4096, 4096); payload[o] = K.encode(vertices=np.asarray(mv, np.float64), faces=np.asarray(mf, np.float64))
for name, Wc in paths.items():
    sol, sub = [], []
    for slot, o in enumerate(objs):
        Tw = np.einsum('nij,njk->nik', Wc[:m], Tco[o][:m]); g = K.pose_cols(gts[o][:m]); oc = K.pose_cols(Tw)
        for f in range(m):
            rid = f'track_3/e{ep:06d}/w000/f{f:06d}/o{slot}'
            sol.append(dict(row_id=rid, episode_index=ep, world_index=0, frame_index=f, object_slot=slot, object_name=o, **{'ref_' + k: float(v[f]) for k, v in g.items()}, reference_geometry_b64=''))
            sub.append(dict(row_id=rid, **{k: float(v[f]) for k, v in oc.items()}, **K.PROV, object_mesh_b64=payload[o] if f == 0 else ''))
    sol = pd.DataFrame(sol); sub = pd.DataFrame(sub); sol.loc[0, 'reference_geometry_b64'] = geom
    sc = {nm: float(md.score(sol.copy(), sub.copy(), 'row_id')) for nm, md in mods.items()}; err = K.aligned_errors(mods['MP-SR'], sol, sub)
    # GT-free static cue for this path vs slam (jitter of resting objects)
    cue = {}
    for o in objs:
        Ts = np.einsum('nij,njk->nik', c2w[:m], Tco[o][:m]); Tx = np.einsum('nij,njk->nik', Wc[:m], Tco[o][:m]); mk_ = rest_mask(Ts, Tx); cue[o] = [jitter(Ts, mk_)[0], jitter(Tx, mk_)[0], int(mk_.sum())]
    res[name] = dict(scores=sc, objects={o: [float(err[o][0]), float(err[o][1])] for o in objs}, static_cue_slam_vs_this=cue)
    print(f'ep{ep} {name:11s} ' + ' '.join(f'{o}: {err[o][0]:6.2f} cm/{err[o][1]:6.1f} deg' for o in objs) + f' | AUC {sc["AUC"]:.3f} MP {sc["MP-SR"]:.0f} MPPE {sc["MPPE"]:.2f}', flush=True)
print('camera', {k: round(v, 2) for k, v in res['camera'].items()}, 'scale fit', round(s, 3))
os.makedirs(O + '/public/compare', exist_ok=True); json.dump(res, open(f'{O}/public/compare/ep{ep}.json', 'w'), indent=1, default=float)
