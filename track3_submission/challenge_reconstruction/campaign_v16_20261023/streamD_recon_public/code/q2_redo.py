"""Q2 redo: displacement-based static window for the anchor, window-mean anchor, GT-oracle anchor; plus boards PnP-camera variant baseline."""
import json, numpy as np, common as C, pp as PP
from scipy.spatial.transform import Rotation as R
OUT = '/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamD_recon_public/results'
def static_k(T, c, thr=0.02, min_len=5):
    cen = np.einsum('nij,j->ni', T[:, :3, :3], c) + T[:, :3, 3]; c0 = np.median(cen[:5], 0); far = np.flatnonzero(np.linalg.norm(cen - c0, axis=1) > thr)
    k = int(far[0]) if len(far) else len(T); return k if k >= min_len else 0
res = {}
for ep in C.CLIPS:
    tr, c2w = C.final_tracks(ep); sc = C.Scene(ep, tr); a = sc.anchor; e = sc.evaluate()[0]
    tw, c, w = PP.principal_twins(C.MESH(sc.names[a])); k = static_k(tr[sc.names[a]], c)
    r = dict(anchor=sc.names[a], static_frames=k, base={o: list(v) for o, v in e.items()})
    if k:
        Tk = np.tile(np.eye(4), (k, 1, 1)); Tk[:, :3, :3] = R.from_quat(sc.ach[:k, 0, a, 3:]).as_matrix(); Tk[:, :3, 3] = sc.ach[:k, 0, a, :3]; m = C.mean_pose(Tk)
        ach2 = sc.ach.copy(); ach2[0, 0, a, :3] = m[:3, 3]; ach2[0, 0, a, 3:] = R.from_matrix(m[:3, :3]).as_quat()
        r['window_mean_anchor'] = {o: list(v) for o, v in sc.evaluate(ach2)[0].items()}
        r['frame0_vs_window_cm_deg'] = [float(np.linalg.norm(sc.ach[0, 0, a, :3] - m[:3, 3]) * 100), float(np.degrees((R.from_quat(sc.ach[0, 0, a, 3:]) * R.from_matrix(m[:3, :3]).inv()).magnitude()))]
    ach3 = sc.ach.copy(); ach3[0, 0, a] = sc.ref[0, 0, a]; r['oracle_gt_frame0_anchor'] = {o: list(v) for o, v in sc.evaluate(ach3)[0].items()}
    # how wrong is our anchor frame 0 after a best-fit (Umeyama, rotation+translation) alignment of the whole anchor track? -> anchor frame-0 residual
    al = sc.M._align_registered_scene(sc.ach, sc.ref, sc.axes)
    P = al[:, 0, a, :3]; Q = sc.ref[:, 0, a, :3]; mp, mq = P.mean(0), Q.mean(0); U, S, Vt = np.linalg.svd((P - mp).T @ (Q - mq)); D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]); Rf = (U @ D @ Vt).T
    Pf = (P - mp) @ Rf.T + mq; r['anchor_track_bestfit_mean_cm'] = float(np.linalg.norm(Pf - Q, axis=1).mean() * 100); r['anchor_frame0_residual_after_bestfit_cm'] = float(np.linalg.norm(Pf[0] - Q[0]) * 100)
    res[ep] = r; print(ep, json.dumps(r), flush=True)
json.dump(res, open(OUT + '/q2_redo.json', 'w'), indent=1)
# boards: PnP-camera final recipe baselines
b = {}
for ep in (23, 24):
    for tag in ('final_tracks_compose_pnp', 'final_tracks_compose_ego', 'final_tracks'):
        z = np.load(f'{C.W}/boards/episode_{ep:06d}/{tag}.npz'); names = [str(x) for x in z['names']]
        tr = {o: z['world_from_object'][:, names.index(o)].astype(float) for o in names}; sc = C.Scene(ep, tr); b[f'{ep}:{tag}'] = {o: list(v) for o, v in sc.evaluate()[0].items()}
        print(ep, tag, b[f'{ep}:{tag}'], flush=True)
json.dump(b, open(OUT + '/boards_variants.json', 'w'), indent=1); print('done')
