"""G17 shared GT-free helpers: episode inputs (masks, stereo depth, support plane), mesh render (silhouette fillPoly,
point-splat z-buffer), SE(3) utils, retime shift. No reference data is read anywhere in this module."""
import json, numpy as np, cv2, trimesh
from pathlib import Path
from scipy.spatial.transform import Rotation as Rot

REC = Path('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026')
CAN = REC / 'canonical'
G15 = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG15')
TUNE = [11, 12, 18, 21, 23, 24, 31, 39, 41, 44]
HELD = [0, 2, 13, 22, 26, 27, 40, 42, 43, 45]
EVAL = [1, 3, 5, 6, 8, 10, 14, 15, 16, 19, 20, 25, 28, 30, 32, 33, 35, 36, 37, 38]
DELTA = json.loads((G15 / 'out/sync_delta_t.json').read_text())


def eproot(split, ep):
    return CAN / {'public': 'public' if ep in TUNE else 'public_heldout', 'evaluation': 'evaluation'}[split] / f'episode_{ep:06d}'


def key(split, ep):
    return f'{split}_{ep}'


def shift(T, d):
    """Resample a [F,4,4] track at t+d (G15 retime.shift)."""
    F = len(T); d = np.broadcast_to(np.asarray(d, float), (F,)) if np.ndim(d) else d
    t = np.clip(np.arange(F) + d, 0, F - 1); i0 = np.floor(t).astype(int); i1 = np.minimum(i0 + 1, F - 1); a = t - i0
    out = T.copy(); out[:, :3, 3] = (1 - a)[:, None] * T[i0, :3, 3] + a[:, None] * T[i1, :3, 3]
    R0 = Rot.from_matrix(T[i0, :3, :3]); R1 = Rot.from_matrix(T[i1, :3, :3]); rv = (R0.inv() * R1).as_rotvec()
    out[:, :3, :3] = (R0 * Rot.from_rotvec(rv * a[:, None])).as_matrix(); return out


def delta_arr(split, ep, F):
    d = DELTA[key(split, ep)]
    return np.asarray(d, float)[:F] if np.ndim(d) else float(d)


def se3(rv, t):
    T = np.eye(4); T[:3, :3] = Rot.from_rotvec(rv).as_matrix(); T[:3, 3] = t; return T


def about(c, rv, t):
    """World-frame correction rotating by rv about point c then translating by t."""
    R = Rot.from_rotvec(rv).as_matrix(); T = np.eye(4); T[:3, :3] = R; T[:3, 3] = c + t - R @ c; return T


class Episode:
    def __init__(self, root, ds=2):
        self.root = Path(root); intr = json.loads((self.root / 'intrinsics.json').read_text()); self.ds = ds
        self.K = np.array([[intr['fx'] / ds, 0, intr['cx'] / ds], [0, intr['fy'] / ds, intr['cy'] / ds], [0, 0, 1.]])
        self.H, self.W = intr['height'] // ds, intr['width'] // ds
        self._m = {}

    def mask(self, obj, lab, f):
        k = (obj, lab, f)
        if k not in self._m:
            m = cv2.imread(str(self.root / 'objects' / obj / 'masks' / str(lab) / f'{f:06d}.png'), 0)
            self._m[k] = None if m is None else cv2.resize((m > 0).astype(np.uint8), (self.W, self.H), interpolation=cv2.INTER_NEAREST) > 0
        return self._m[k]

    def hands(self, objs, f):
        h = np.zeros((self.H, self.W), bool)
        for o in objs:
            for lab in (2, 3):
                q = self.mask(o, lab, f)
                if q is not None: h |= q
        return h

    def depth(self, f):
        enc = cv2.imread(str(self.root / 'depth' / f'{f:06d}.png'), -1)
        if enc is None: return None
        d = 65535 / np.maximum(enc.astype(float), 1) - 1; d[(enc == 0) | (enc >= 65535)] = np.nan
        return cv2.resize(d, (self.W, self.H), interpolation=cv2.INTER_NEAREST)

    def plane_world(self, cam_bc, frames):
        """Table plane (n, d) in world, n.x + d = 0, n up (towards the camera), median over frames (stereo clock camera)."""
        z = np.load(self.root / 'support_plane.npz'); N, D = [], []
        for f in frames:
            if f >= len(z['plane_observed']) or not z['plane_observed'][f]: continue
            n = z['plane_normals_camera'][f]; d = z['plane_constants_camera'][f]   # plane n.x = d (camera frame)
            if -d < 0: n, d = -n, -d   # camera origin on the positive (up) side
            R, t = cam_bc[f, :3, :3], cam_bc[f, :3, 3]; nw = R @ n; dw = -(d + nw @ t)   # height = nw.x + dw
            N.append(nw); D.append(dw)
        if not N: return None
        n = np.median(N, 0); n /= np.linalg.norm(n); return n, float(np.median(D))


class Mesh:
    def __init__(self, path, faces=4000, pts=5000):
        m = trimesh.load(str(path), force='mesh', process=False)
        self.V = np.asarray(m.vertices, np.float64)
        ms = m.simplify_quadric_decimation(face_count=faces) if len(m.faces) > faces else m
        self.Vr, self.Fr = np.asarray(ms.vertices, np.float64), np.asarray(ms.faces)
        self.P, _ = trimesh.sample.sample_surface(m, pts, seed=1); self.P = np.asarray(self.P)
        self.c = self.V.mean(0)
        hull = m.convex_hull; self.Vh = np.asarray(hull.vertices, np.float64)


def project(K, P):
    return P[:, :2] / P[:, 2:] @ K[:2, :2].T + K[:2, 2]


def render_sil(ep, mesh, T):
    P = mesh.Vr @ T[:3, :3].T + T[:3, 3]
    if (P[:, 2] < .05).any(): return None
    uv = project(ep.K, P); img = np.zeros((ep.H, ep.W), np.uint8)
    cv2.fillPoly(img, list(np.rint(uv[mesh.Fr] * 4).astype(np.int32)), 1, lineType=cv2.LINE_8, shift=2)
    return img > 0


def render_depth(ep, mesh, T):
    P = mesh.P @ T[:3, :3].T + T[:3, 3]
    if (P[:, 2] < .05).any(): return None
    u = np.rint(project(ep.K, P)).astype(int); ok = (u[:, 0] >= 0) & (u[:, 0] < ep.W) & (u[:, 1] >= 0) & (u[:, 1] < ep.H)
    zb = np.full((ep.H, ep.W), np.inf); np.minimum.at(zb, (u[ok, 1], u[ok, 0]), P[ok, 2])
    zb = -cv2.dilate(-np.where(np.isfinite(zb), zb, 1e3), np.ones((3, 3)))  # close splat holes
    zb[zb > 999] = np.inf; return zb


def load_poses(npz):
    z = dict(np.load(npz)); return z, [str(n) for n in z['names']]
