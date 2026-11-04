"""Marker-ball refinement of an object track (input video only; no ground truth).

Every scored object carries small light-grey mocap balls on black mounts. Given an initial object->camera track
(T_init[k]), the undistorted RGB video with its intrinsics K (full resolution) and the object mesh:
  1. learn the ball layout in BODY coordinates from the first `learn` frames: detect grey blobs near the projected
     object, cast their camera rays into the body frame with T_init, cluster them, and intersect each cluster's rays
     (least squares). Optionally use stereo depth (half resolution) to seed the clustering.
  2. track: per frame, predict ball pixels from the previous camera-frame correction applied to T_init[k], detect
     grey blobs in gates around the predictions, and solve the 6-DoF pose by robust least squares on the ball
     reprojection error plus a weak prior toward T_init[k] (keeps roll/depth defined with only 1-2 balls).
  3. smooth the BODY-frame correction C[k] = T_init[k]^-1 T_ref[k] over time and output T_init[k] C_s[k]
     (a wrong roll or offset of the initial track is constant in the body frame, so it is carried across frames).
"""
import numpy as np, cv2
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R


def se3(rv, t):
    T = np.eye(4); T[:3, :3] = R.from_rotvec(rv).as_matrix(); T[:3, 3] = t; return T


def project(K, Pc):
    uv = Pc @ K.T; return uv[:, :2] / uv[:, 2:3]


def grey_blobs(img, roi, r_px, vmax=235, dark_ring=0.0):
    """Light-grey, low-saturation, roughly round blobs inside roi that sit next to a black mount (dark pixels in a ring
    around the blob). Returns (N,2) centres, (N,) radii, (N,) dark-ring fraction."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s, v = hsv[..., 1].astype(np.int16), hsv[..., 2].astype(np.int16)
    m = ((s < 45) & (v > 100) & (v < vmax)).astype(np.uint8) & roi
    dark = (v < 70).astype(np.uint8)
    k = max(1, int(round(r_px * 0.25)))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m, 8)
    out = []
    amin, amax = np.pi * (0.45 * r_px) ** 2, np.pi * (2.0 * r_px) ** 2
    H, W = v.shape
    for i in range(1, n):
        a = st[i, cv2.CC_STAT_AREA]
        if not (amin <= a <= amax): continue
        w, h = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
        if max(w, h) > 2.3 * min(w, h): continue
        if a / float(w * h) < 0.5: continue
        cx, cy = cen[i]; rr = np.sqrt(a / np.pi)
        x0, x1 = int(max(0, cx - 2.2 * rr)), int(min(W, cx + 2.2 * rr + 1)); y0, y1 = int(max(0, cy - 2.2 * rr)), int(min(H, cy + 2.2 * rr + 1))
        yy, xx = np.mgrid[y0:y1, x0:x1]; d = np.hypot(xx - cx, yy - cy); ring = (d > 1.15 * rr) & (d < 2.1 * rr)
        frac = float(dark[y0:y1, x0:x1][ring].mean()) if ring.any() else 0.0
        if frac < dark_ring: continue
        out.append((cx, cy, rr, frac))
    if not out: return np.zeros((0, 2)), np.zeros(0), np.zeros(0)
    o = np.asarray(out); return o[:, :2], o[:, 2], o[:, 3]


def silhouette_roi(K, Tco, V, shape, margin_px):
    uv = project(K, V @ Tco[:3, :3].T + Tco[:3, 3])
    roi = np.zeros(shape[:2], np.uint8)
    ok = np.isfinite(uv).all(1)
    if ok.sum() < 3: return roi
    hull = cv2.convexHull(np.round(uv[ok]).astype(np.int32)); cv2.fillConvexPoly(roi, hull, 1)
    if margin_px > 0: roi = cv2.dilate(roi, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin_px + 1, 2 * margin_px + 1)))
    return roi


def ball_radius_px(K, z, ball_r_m=0.0065):
    return float(K[0, 0] * ball_r_m / max(z, 0.15))


def near_surface(K, Tco, V, cen, k, depth_fn, max_m=0.03):
    """Keep blobs whose stereo-depth point lies within max_m of the posed mesh (rejects background bolts/floor)."""
    if depth_fn is None or not len(cen): return np.ones(len(cen), bool)
    from scipy.spatial import cKDTree
    tree = cKDTree(V @ Tco[:3, :3].T + Tco[:3, 3]); keep = np.zeros(len(cen), bool)
    for i, (u, v) in enumerate(cen):
        z = depth_fn(k, u, v)
        if z is None: keep[i] = True; continue      # no depth: do not veto
        d = np.linalg.solve(K, [u, v, 1.0]); p = d / d[2] * (z + 0.006)
        keep[i] = tree.query(p)[0] < max_m
    return keep


class FrameReader:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(str(path)); self.i = -1; self.frame = None
    def get(self, k):
        if k != self.i + 1: self.cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        ok, f = self.cap.read(); self.i = k
        if not ok: raise IOError(f'frame {k}')
        return f


def learn_layout(reader, K, T_init, V, frames, depth_fn=None, min_support=0.3, cluster_m=0.012, max_surface_m=0.035):
    """Ball centres in body coordinates from rays of detections in `frames` (T_init assumed good there)."""
    obs = []   # (frame, ray origin body, ray dir body, depth-lifted body point or None)
    for k in frames:
        img = reader.get(k); Tco = T_init[k]; zc = float(np.median((V @ Tco[:3, :3].T + Tco[:3, 3])[:, 2]))
        r = ball_radius_px(K, zc); roi = silhouette_roi(K, Tco, V, img.shape, int(3.5 * r + 8))
        c, rad, sc = grey_blobs(img, roi, r)
        kp = near_surface(K, Tco, V, c, k, depth_fn); c, rad = c[kp], rad[kp]
        Toc = np.linalg.inv(Tco)
        for (u, v), rr in zip(c, rad):
            d = np.linalg.solve(K, [u, v, 1.0]); d /= np.linalg.norm(d)
            o_b = Toc[:3, 3]; d_b = Toc[:3, :3] @ d
            p = None
            if depth_fn is not None:
                zd = depth_fn(k, u, v)
                if zd is not None: p = Toc[:3, :3] @ (d / d[2] * (zd + 0.006)) + Toc[:3, 3]   # +ball radius: depth hits the front
            if p is None:   # project onto the closest point of the ray to the body origin region (surface prior)
                s = max(0.05, float(-(o_b @ d_b))); p = o_b + s * d_b
            obs.append((k, o_b, d_b, p))
    if not obs: return np.zeros((0, 3)), []
    P = np.array([o[3] for o in obs])
    # greedy clustering of seed points
    lab = -np.ones(len(P), int); cid = 0
    order = np.argsort([np.linalg.norm(p) for p in P])
    for i in order:
        if lab[i] >= 0: continue
        d = np.linalg.norm(P - P[i], axis=1); m = (d < 2.5 * cluster_m) & (lab < 0); lab[m] = cid; cid += 1
    balls, info = [], []
    nfr = len(set(frames))
    for c in range(cid):
        idx = np.where(lab == c)[0]; fr = {obs[i][0] for i in idx}
        if len(fr) < max(3, min_support * nfr): continue
        # least-squares ray intersection
        A = np.zeros((3, 3)); b = np.zeros(3)
        for i in idx:
            o, d = obs[i][1], obs[i][2]; M = np.eye(3) - np.outer(d, d); A += M; b += M @ o
        x = np.linalg.lstsq(A, b, rcond=None)[0]
        # robust re-fit: drop rays far from x
        keep = [i for i in idx if np.linalg.norm((np.eye(3) - np.outer(obs[i][2], obs[i][2])) @ (x - obs[i][1])) < 0.01]
        if len(keep) >= 3:
            A = np.zeros((3, 3)); b = np.zeros(3)
            for i in keep:
                o, d = obs[i][1], obs[i][2]; M = np.eye(3) - np.outer(d, d); A += M; b += M @ o
            x = np.linalg.lstsq(A, b, rcond=None)[0]
        dsurf = float(np.min(np.linalg.norm(V - x, axis=1)))
        if dsurf > max_surface_m: continue
        balls.append(x); info.append(dict(support_frames=len(fr), rays=len(idx), kept=len(keep), surface_cm=round(dsurf * 100, 2)))
    return np.array(balls), info


def solve_pose(K, Tpred, Tinit, balls, det, sig_px=3.0, sig_rot_deg=25.0, sig_t_m=0.05):
    """Robust LS on reprojection of matched balls + weak prior to Tinit. det: list of (ball_index, (u,v))."""
    bi = np.array([d[0] for d in det]); uv = np.array([d[1] for d in det], float)
    Tinv = np.linalg.inv(Tinit)
    def f(x):
        T = se3(x[:3], x[3:]) @ Tpred; Pc = balls[bi] @ T[:3, :3].T + T[:3, 3]
        r1 = ((project(K, Pc) - uv) / sig_px).ravel()
        D = T @ Tinv; r2 = R.from_matrix(D[:3, :3]).as_rotvec() / np.radians(sig_rot_deg); r3 = D[:3, 3] / sig_t_m
        return np.concatenate([r1, r2, r3])
    s = least_squares(f, np.zeros(6), loss='soft_l1', f_scale=2.0)
    T = se3(s.x[:3], s.x[3:]) @ Tpred
    Pc = balls[bi] @ T[:3, :3].T + T[:3, 3]; res = np.linalg.norm(project(K, Pc) - uv, axis=1)
    return T, res


def track(reader, K, T_init, V, balls, frames, gate_px=None, min_balls=2, max_res_px=8.0, depth_fn=None):
    n = len(T_init); C = np.eye(4); out = np.array(T_init, float).copy(); stats = []
    for k in frames:
        img = reader.get(k); Tpred = T_init[k] @ C
        Pc = balls @ Tpred[:3, :3].T + Tpred[:3, 3]; pred = project(K, Pc); z = float(np.median(Pc[:, 2]))
        r = ball_radius_px(K, z); g = gate_px or max(18.0, 3.0 * r)
        roi = silhouette_roi(K, Tpred, np.vstack([V, balls]), img.shape, int(g + 3 * r))
        cen, rad, sc = grey_blobs(img, roi, r)
        kp = near_surface(K, Tpred, np.vstack([V, balls]), cen, k, depth_fn, 0.04); cen = cen[kp]
        det = []
        if len(cen):
            used = set()
            for i, p in enumerate(pred):
                d = np.linalg.norm(cen - p, axis=1); j = int(np.argmin(d))
                if d[j] < g and j not in used: det.append((i, cen[j])); used.add(j)
        st = dict(frame=k, n_det=len(det), accepted=False)
        if len(det) >= min_balls:
            T, res = solve_pose(K, Tpred, T_init[k], balls, det)
            good = res < max_res_px
            if good.sum() >= min_balls and good.sum() < len(det):
                T, res = solve_pose(K, Tpred, T_init[k], balls, [d for d, gg in zip(det, good) if gg])
            if (res < max_res_px).sum() >= min_balls:
                out[k] = T; C = np.linalg.inv(T_init[k]) @ T; st.update(accepted=True, res_px=float(np.median(res)))
        if not st['accepted']:
            out[k] = Tpred
        st['corr_deg'] = float(np.degrees(R.from_matrix(C[:3, :3]).magnitude())); st['corr_cm'] = float(np.linalg.norm(C[:3, 3]) * 100)
        stats.append(st)
    return out, stats


def smooth_correction(T_out, T_init, half=3):
    n = len(T_out); C = np.array([np.linalg.inv(T_init[k]) @ T_out[k] for k in range(n)])
    rv = R.from_matrix(C[:, :3, :3]); res = np.array(T_init, float).copy()
    for k in range(n):
        a, b = max(0, k - half), min(n, k + half + 1)
        rm = R.from_matrix(C[a:b, :3, :3]).mean(); t = C[a:b, :3, 3].mean(0)
        Cs = np.eye(4); Cs[:3, :3] = rm.as_matrix(); Cs[:3, 3] = t; res[k] = T_init[k] @ Cs
    return res
