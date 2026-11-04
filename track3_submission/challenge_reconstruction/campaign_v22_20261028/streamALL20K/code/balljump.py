"""ALL20-K (GT-free) rigid-ball consistency audit for every object of a bundle. The marker balls are rigidly attached, so OUR stereo
ball detections expressed in the object frame (via the bundle's camera_from_object) must form the same fixed clusters in every
frame. A pose error E confined to a window (e.g. the static start, as in ep35/ep33) shows as that window's object-frame clusters
being rigidly moved vs the rest: q_win = E^-1 b. Kabsch(rest <- window) over shared clusters estimates E (object frame).
Fix would be T_true = T_win @ inv(E). Also reports per-window scatter (tracking noise) and the roll about the mesh long PCA axis.
usage: balljump.py BUNDLE_EP_DIR EP OUT.json"""
import sys, json, numpy as np, trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
B, EP, OUT = sys.argv[1], int(sys.argv[2]), sys.argv[3]
C = f'/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/canonical/evaluation/episode_{EP:06d}'
z = np.load(f'{B}/object_poses.npz', allow_pickle=False); names = [str(n) for n in z['names']]
Tc_all, Tw_all = z['camera_from_object'], z['world_from_object']

def kabsch(A, Bp):   # R,t with Bp ~ A @ R.T + t
    a, b = A.mean(0), Bp.mean(0); H = (A - a).T @ (Bp - b); U, s, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]); R = Vt.T @ D @ U.T; return R, b - R @ a

def clusters(Q, rad=0.012, minn=8):
    left = np.ones(len(Q), bool); out = []; tree = cKDTree(Q)
    cnt = np.array([len(x) for x in tree.query_ball_point(Q, rad)])
    while True:
        idx = np.where(left)[0]
        if not len(idx): break
        i = idx[np.argmax(cnt[idx])]
        c = Q[i]
        for _ in range(5):
            nb = [j for j in tree.query_ball_point(c, rad) if left[j]]
            if not nb: break
            c = np.median(Q[nb], 0)
        nb = [j for j in tree.query_ball_point(c, 1.25 * rad) if left[j]]
        if len(nb) < minn: break
        out.append((c, nb)); left[nb] = False
    return out

def windows(Tw):
    p = Tw[:, :3, 3]; R = Tw[:, :3, :3]; n = len(Tw)
    # static = within 1.5 cm / 12 deg of the end frame pose; a window ends at the first of 3 consecutive "moved" frames
    def moved(t, ref): return np.linalg.norm(p[t] - p[ref]) > 0.015 or np.degrees(Rot.from_matrix(R[t] @ R[ref].T).magnitude()) > 12
    s = 0
    while s + 3 < n and not (moved(s + 1, 0) and moved(s + 2, 0) and moved(s + 3, 0)): s += 1
    e = n - 1
    while e - 3 > s and not (moved(e - 1, n - 1) and moved(e - 2, n - 1) and moved(e - 3, n - 1)): e -= 1
    return s, e   # start static = [0, s], end static = [e, n-1]

res = {}
for k, name in enumerate(names):
    m = trimesh.load(f'{B}/meshes/{name}.obj', force='mesh'); v = np.asarray(m.vertices)
    ctr = v.mean(0); rad = np.linalg.norm(v - ctr, axis=1).max() + 0.03
    c0 = v - ctr; w, U = np.linalg.eigh(c0.T @ c0 / len(c0)); long = U[:, 2]
    if long[np.argmax(np.abs(long))] < 0: long = -long
    try: fr = json.load(open(f'{C}/ball_detections_{name}.json'))['frames']
    except FileNotFoundError: res[name] = dict(error='no detections'); continue
    Tc = Tc_all[:, k]; n = min(len(Tc), len(fr)); Q, F = [], []
    for t in range(n):
        if not fr[t]: continue
        D = np.array([d['camera_point'] for d in fr[t]]); cc = Tc[t][:3, :3] @ ctr + Tc[t][:3, 3]
        D = D[np.linalg.norm(D - cc, axis=1) < rad]
        if not len(D): continue
        q = (D - Tc[t][:3, 3]) @ Tc[t][:3, :3]; Q.append(q); F += [t] * len(q)
    if not Q: res[name] = dict(error='no inlier detections'); continue
    Q = np.concatenate(Q); F = np.array(F)
    s, e = windows(Tw_all[:, k])
    cl = clusters(Q)
    lab = np.full(len(Q), -1)
    for ci, (c, nb) in enumerate(cl): lab[nb] = ci
    wins = {'start': (F <= s), 'end': (F >= e), 'mid': (F > s) & (F < e)}
    out = dict(n_frames=n, start_static=[0, int(s)], end_static=[int(e), n - 1], n_points=int(len(Q)), n_clusters=len(cl),
               cluster_sizes=[len(nb) for _, nb in cl], frames_with_balls={w: int(len(set(F[msk]))) for w, msk in wins.items()})
    def med(msk, ci):
        sel = msk & (lab == ci); return (np.median(Q[sel], 0), int(sel.sum()), float(np.median(np.linalg.norm(Q[sel] - np.median(Q[sel], 0), axis=1))) * 100) if sel.sum() >= 4 else None
    for wn in ('start', 'end', 'mid'):
        other = ~wins[wn]
        pairs = [(med(wins[wn], ci), med(other, ci)) for ci in range(len(cl))]
        pairs = [(a, b) for a, b in pairs if a and b]
        info = dict(shared_clusters=len(pairs), scatter_cm=[round(a[2], 2) for a, b in pairs], npts=[a[1] for a, b in pairs])
        if len(pairs) >= 3:
            A = np.array([a[0] for a, b in pairs]); Bq = np.array([b[0] for a, b in pairs])
            Rm, tm = kabsch(A, Bq)   # rest ~ R win + t  -> E
            rv = Rot.from_matrix(Rm).as_rotvec(); ang = np.degrees(np.linalg.norm(rv))
            resid = np.linalg.norm(A @ Rm.T + tm - Bq, axis=1) * 100
            # effect at the object: max displacement of mesh vertices under E
            disp = np.linalg.norm((v @ Rm.T + tm) - v, axis=1) * 100
            info.update(E_rot_deg=round(ang, 2), E_axis_obj=np.round(rv / max(np.linalg.norm(rv), 1e-9), 3).tolist(),
                        E_roll_about_long_deg=round(float(np.degrees(rv @ long)), 2), E_trans_cm=round(float(np.linalg.norm(tm) * 100), 2),
                        kabsch_resid_cm=np.round(resid, 2).tolist(), mean_vertex_disp_cm=round(float(disp.mean()), 2))
        elif len(pairs) == 2:
            a0, a1 = pairs[0][0][0], pairs[1][0][0]; b0, b1 = pairs[0][1][0], pairs[1][1][0]
            da, db = a1 - a0, b1 - b0
            info.update(pair_dir_change_deg=round(float(np.degrees(np.arccos(np.clip(da @ db / np.linalg.norm(da) / np.linalg.norm(db), -1, 1)))), 2),
                        pair_mid_shift_cm=round(float(np.linalg.norm((a0 + a1) / 2 - (b0 + b1) / 2) * 100), 2))
        out[wn] = info
    res[name] = out
json.dump(res, open(OUT, 'w'), indent=1)
for n_, r in res.items():
    if 'error' in r: print(EP, n_, r['error']); continue
    print(EP, n_, 'clusters', r['n_clusters'], 'static', r['start_static'], r['end_static'], 'balls/frames', r['frames_with_balls'])
    for wn in ('start', 'end', 'mid'):
        w = r[wn]; print('   ', wn, {k: w[k] for k in w if k not in ('npts',)})
