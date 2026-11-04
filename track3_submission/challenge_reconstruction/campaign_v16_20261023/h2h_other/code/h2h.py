"""Per-frame head-to-head evidence for one episode: for every object and both tracks (each with its own paired mesh), in
the physical ego_cam_a frame: stereo depth residual (data->model, masked object points, hand pixels removed, 3 px erosion),
silhouette IoU vs the SAM2 object mask (hand pixels ignored), and the track-vs-track difference (posed-surface centroid
distance; rotation through the kit registration of each mesh to the public scan, symmetry-aware).
Also: a world-offset-tolerant residual: one constant translation per track and episode in that track's own world (shared by
all objects; the official frame-0 alignment removes such an offset) fitted to the masked stereo points, then re-scored.
usage: h2h.py EP STRIDE OUT.json"""
import sys, os, json, importlib.util, numpy as np, cv2, trimesh
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, os.path.dirname(__file__)); from common import *
KIT = '/data1/SRL_V2D/challenge_submissions_SRL/track3_v13_20261022/all_20_episodes/submission_kit'
spec = importlib.util.spec_from_file_location('mpsr', KIT + '/metric_code/track_3/MP-SR.py'); M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
SYM = {'beige_cup': 0, 'blue_cup': 0, 'white_pot_lid': 2}   # 0 = continuous about the axis, 2 = half turn
def scan(n):
    m = trimesh.load(SCANS / f'{n}.ply', force='mesh', process=False); return np.asarray(m.vertices, float), np.asarray(m.faces, np.int64)
def sym_axis(v):
    w, U = np.linalg.eigh(np.cov((v - v.mean(0)).T)); gaps = [abs(w[0] - w[1]) + abs(w[0] - w[2]), abs(w[1] - w[0]) + abs(w[1] - w[2]), abs(w[2] - w[0]) + abs(w[2] - w[1])]
    return U[:, int(np.argmax(gaps))]
def registration(mesh, n, ep, slot):
    """kit _register_object_body (exact scorer seed): returns body_from_scan 4x4 placement (scan centroid frame -> body)."""
    rv, rf = scan(n); cv, cf = np.asarray(mesh.vertices, float), np.asarray(mesh.faces, np.int64); rc, rr = M._surface_moments(rv, rf)
    anchor, rot = M._register_object_body((cv, cf), (rv, rf), rc, 1_000_003 * ep + 101 * slot)
    P = np.eye(4); P[:3, :3] = rot.T; P[:3, 3] = anchor; return P   # scan-centred frame (scan axes, origin at scan centroid)
def rot_diff(Ra, Rb, axis, order):
    if order == 0:
        a, b = Ra @ axis, Rb @ axis; return np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b)), np.clip(a @ b, -1, 1)))
    d = np.degrees(np.linalg.norm(R.from_matrix(Ra.T @ Rb).as_rotvec()))
    if order == 2:
        S = R.from_rotvec(np.pi * axis).as_matrix(); d = min(d, np.degrees(np.linalg.norm(R.from_matrix((Ra @ S).T @ Rb).as_rotvec())))
    return d
def twin_set(v):
    w, U = np.linalg.eigh(np.cov((v - v.mean(0)).T)); return [np.eye(3)] + [R.from_rotvec(np.pi * U[:, k]).as_matrix() for k in range(3)]
if __name__ == '__main__':
    ep, stride, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    o = load_ours(ep); n = load_new(ep, o['names']); names = n['names']
    override = None
    if ep == 25:
        B = CR / 'public_gt_check_20261020/boards/episode_000025'; override = {'wooden_piece_1': B / 'A/masks', 'wooden_piece_2': B / 'B/masks'}
    ev = Evidence(ep, o['names'], override)
    trk = {'ours': ({k: o['A_T_obj'][:, o['names'].index(k)] for k in names}, o['meshes']), 'new': ({k: n['A_T_obj'][k] for k in names}, n['meshes'])}
    MD = {(t, k): MeshDist(trk[t][1][k]) for t in trk for k in names}; SIL = {(t, k): sil_mesh(trk[t][1][k]) for t in trk for k in names}
    WC = {'ours': o['world_from_cam'], 'new': np.einsum('tij,jk->tik', n['cam0_T_cam'], np.linalg.inv(A_T_rect(ep)))}; STORE = []
    REG = {}; CEN = {}
    for t in trk:
        for k in names:
            REG[(t, k)] = registration(trk[t][1][k], k, ep, o['names'].index(k)); CEN[(t, k)] = trimesh.Trimesh(trk[t][1][k].vertices, trk[t][1][k].faces).area_faces @ trk[t][1][k].triangles_center / trk[t][1][k].area
    AX = {k: sym_axis(scan(k)[0]) for k in names}; TW = {k: twin_set(scan(k)[0]) for k in names}
    T = min(len(o['A_T_obj']), *[len(n['A_T_obj'][k]) for k in names]); rows = []
    ker = np.ones((7, 7), np.uint8)
    for t in range(0, T, stride):
        P, uv = ev.stereo_points(t); hand = ev.hands(t); row = dict(frame=t)
        hand_small = cv2.resize(hand.astype(np.uint8), (ev.size[0] // 2, ev.size[1] // 2), interpolation=cv2.INTER_NEAREST) > 0
        for k in names:
            m = ev.mask(k, t); r = {}
            if m is not None and P is not None:
                me = (cv2.erode(m.astype(np.uint8), ker) > 0) & ~(cv2.dilate(hand.astype(np.uint8), ker) > 0)
                ui = np.clip(np.rint(uv).astype(int), 0, [ev.size[0] - 1, ev.size[1] - 1]); sel = me[ui[:, 1], ui[:, 0]]; Q = P[sel]
                if len(Q) > 5000: Q = Q[np.random.default_rng(t).choice(len(Q), 5000, replace=False)]
                r['n_pts'] = int(len(Q)); r['mask_px'] = int(m.sum())
                if len(Q) >= 40: STORE.append((t, k, Q[:1500]))
                ms = cv2.resize(m.astype(np.uint8), hand_small.shape[::-1], interpolation=cv2.INTER_NEAREST) > 0
                for tn in trk:
                    A = trk[tn][0][k][t]; e = {}
                    if len(Q) >= 40:
                        d = np.sort(MD[(tn, k)].d(Q, A)) * 1000; e['med_mm'] = float(np.median(d)); e['trim_mm'] = float(d[:int(.8 * len(d))].mean()); e['in8'] = float((d < 8).mean())
                    s = render_sil(ev, SIL[(tn, k)], A)
                    if s is not None:
                        valid = ~hand_small; e['iou'] = float((s & ms & valid).sum() / max(((s | ms) & valid).sum(), 1))
                        e['spill'] = float((s & ~ms & valid).sum() / max((s & valid).sum(), 1)); e['cover'] = float((s & ms).sum() / max(ms.sum(), 1))
                    r[tn] = e
            # track difference
            Ao, An = trk['ours'][0][k][t], trk['new'][0][k][t]
            co = Ao[:3, :3] @ CEN[('ours', k)] + Ao[:3, 3]; cn = An[:3, :3] @ CEN[('new', k)] + An[:3, 3]
            So = Ao @ REG[('ours', k)]; Sn = An @ REG[('new', k)]
            r['d_cen_cm'] = float(np.linalg.norm(co - cn) * 100); r['d_anchor_cm'] = float(np.linalg.norm(So[:3, 3] - Sn[:3, 3]) * 100)
            r['d_rot_deg'] = float(rot_diff(So[:3, :3], Sn[:3, :3], AX[k], SYM.get(k, -1)))
            r['d_rot_twinmin_deg'] = float(min(np.degrees(np.linalg.norm(R.from_matrix((So[:3, :3] @ G).T @ Sn[:3, :3]).as_rotvec())) for G in TW[k]))
            r['d_rot_twin_idx'] = int(np.argmin([np.linalg.norm(R.from_matrix((So[:3, :3] @ G).T @ Sn[:3, :3]).as_rotvec()) for G in TW[k]]))
            row[k] = r
        rows.append(row)
        print(t, ' | '.join(f"{k[:10]} n{row[k].get('n_pts',0)} d{row[k]['d_cen_cm']:.1f}cm/{row[k]['d_rot_deg']:.0f}deg(tw{row[k]['d_rot_twinmin_deg']:.0f}) " +
              ' '.join(f"{tn}:{row[k].get(tn,{}).get('med_mm',np.nan):.1f}/{row[k].get(tn,{}).get('iou',np.nan):.2f}" for tn in trk) for k in names), flush=True)
    # world-offset-tolerant residual
    offs = {}
    for tn in trk:
        vecs = []
        for (t, k, Q) in STORE:
            A = trk[tn][0][k][t]; loc = (Q - A[:3, 3]) @ A[:3, :3]; d, i = MD[(tn, k)].tree.query(loc)
            v = (MD[(tn, k)].s[i] - loc) @ A[:3, :3].T            # data->model displacement in the camera
            vecs.append(WC[tn][t][:3, :3] @ np.median(v, 0))        # in that track's world
        off = np.median(np.array(vecs), 0) if vecs else np.zeros(3)
        for _ in range(2):   # refine
            vv = []
            for (t, k, Q) in STORE:
                A = trk[tn][0][k][t].copy(); A[:3, 3] -= WC[tn][t][:3, :3].T @ off; loc = (Q - A[:3, 3]) @ A[:3, :3]; d, i = MD[(tn, k)].tree.query(loc)
                vv.append(WC[tn][t][:3, :3] @ np.median((MD[(tn, k)].s[i] - loc) @ A[:3, :3].T, 0))
            off = off + np.median(np.array(vv), 0)
        offs[tn] = off
        for (t, k, Q) in STORE:
            A = trk[tn][0][k][t].copy(); A[:3, 3] -= WC[tn][t][:3, :3].T @ off; d = np.sort(MD[(tn, k)].d(Q, A)) * 1000
            row = next(r for r in rows if r['frame'] == t); row[k].setdefault(tn, {})['med_off_mm'] = float(np.median(d))
    print('world offsets (cm):', {tn: (offs[tn] * 100).round(2).tolist() for tn in offs})
    json.dump(dict(ep=ep, names=names, ours_src=o['src'], world_offset_m={tn: offs[tn].tolist() for tn in offs}, rows=rows), open(out, 'w'))
