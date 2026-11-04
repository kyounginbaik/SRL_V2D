"""W8 nest tools (no GT): container axes, mesh-mesh collision along an axis, joint-occlusion rendering.

All inputs are our own meshes, poses, SAM2 masks and stereo depth of a canonical clip folder.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import trimesh
import fcl
import cv2


def load_mesh(path):
    return trimesh.load(str(path), force='mesh', process=False)


def container_axis(mesh):
    """Principal (longest) axis, signed toward the wider end (the opening of a cup / pot)."""
    v = np.asarray(mesh.vertices); c = v.mean(0)
    w, U = np.linalg.eigh(np.cov((v - c).T)); a = U[:, -1]
    h = (v - c) @ a; r = np.linalg.norm((v - c) - np.outer(h, a), axis=1)
    if r[h > np.percentile(h, 85)].mean() < r[h < np.percentile(h, 15)].mean():
        a = -a
    return a, c, float(w[-1] / max(w[1], 1e-12))


def _bvh(mesh):
    m = fcl.BVHModel(); m.beginModel(len(mesh.vertices), len(mesh.faces))
    m.addSubModel(np.asarray(mesh.vertices, float), np.asarray(mesh.faces, np.int32)); m.endModel(); return m


class Pair:
    def __init__(self, outer, inner):
        self.outer, self.inner = outer, inner
        self.bo, self.bi = _bvh(outer), _bvh(inner)

    def collide(self, To, Ti):
        oo = fcl.CollisionObject(self.bo, fcl.Transform(To[:3, :3], To[:3, 3]))
        oi = fcl.CollisionObject(self.bi, fcl.Transform(Ti[:3, :3], Ti[:3, 3]))
        req = fcl.CollisionRequest(num_max_contacts=1); res = fcl.CollisionResult()
        return fcl.collide(oo, oi, req, res) > 0

    def min_offset(self, To, Ti, axis_world, dmax=0.12, tol=2e-4):
        """Smallest d>=0 such that the inner, translated by d*axis_world, does not intersect the outer.
        Returns 0 if already free; None if still colliding at dmax."""
        def T(d):
            X = Ti.copy(); X[:3, 3] = Ti[:3, 3] + d * axis_world; return X
        if not self.collide(To, Ti):
            return 0.0
        lo, hi = 0.0, None
        d = 0.002
        while d <= dmax:
            if not self.collide(To, T(d)):
                hi = d; break
            lo = d; d *= 1.6
        if hi is None:
            return None
        while hi - lo > tol:
            m = (lo + hi) / 2
            if self.collide(To, T(m)): lo = m
            else: hi = m
        return hi


def intrinsics(root):
    intr = json.loads((Path(root) / 'intrinsics.json').read_text())
    K = np.array([[intr['fx'], 0, intr['cx']], [0, intr['fy'], intr['cy']], [0, 0, 1.]])
    return K, int(intr['width']), int(intr['height'])


class Renderer:
    """Ray-cast z-buffer of several posed meshes in the (undistorted, pinhole) RGB camera."""

    def __init__(self, meshes, K, W, H, step=1):
        self.meshes = meshes; self.K = K; self.W = W; self.H = H; self.step = step
        self.inter = [trimesh.ray.ray_pyembree.RayMeshIntersector(m) for m in meshes]

    def render(self, cam_from_obj, roi=None, bias=None):
        """Return label map (-1 none, i = mesh index) and depth (camera z) over roi=(x0,y0,x1,y1)."""
        x0, y0, x1, y1 = roi if roi is not None else (0, 0, self.W, self.H)
        xs = np.arange(x0, x1, self.step); ys = np.arange(y0, y1, self.step)
        u, v = np.meshgrid(xs + .5 * self.step, ys + .5 * self.step)
        d = np.stack([(u - self.K[0, 2]) / self.K[0, 0], (v - self.K[1, 2]) / self.K[1, 1], np.ones_like(u)], -1).reshape(-1, 3)
        best = np.full(len(d), np.inf); lab = np.full(len(d), -1)
        for i, (inter, T) in enumerate(zip(self.inter, cam_from_obj)):
            Ti = np.linalg.inv(T)
            o = np.tile(Ti[:3, 3], (len(d), 1)); dd = d @ Ti[:3, :3].T
            loc, ri, _ = inter.intersects_location(o, dd, multiple_hits=False)
            if not len(ri): continue
            z = (loc @ T[:3, :3].T + T[:3, 3])[:, 2]
            zb = z + (0 if bias is None else bias[i])
            upd = zb < best[ri]
            best[ri[upd]] = zb[upd]; lab[ri[upd]] = i
        shp = (len(ys), len(xs))
        return lab.reshape(shp), np.where(np.isfinite(best), best, np.nan).reshape(shp), (x0, y0, x1, y1)


def read_mask(root, name, frame, label='1'):
    p = Path(root) / 'objects' / name / 'masks' / label / f'{frame:06d}.png'
    m = cv2.imread(str(p), 0)
    return None if m is None else m > 0


def read_depth(root, frame):
    enc = cv2.imread(str(Path(root) / 'depth' / f'{frame:06d}.png'), -1).astype(float)
    d = 65535 / np.maximum(enc, 1) - 1; d[(enc == 0) | (enc >= 65535)] = np.nan
    return d


def render_nested(rend_outer, rend_inner, T_out, T_in, roi, axis_o, centre_o, rim_h_o, eps=0.015):
    """Label map (-1 none, 0 outer, 1 inner) where the inner object's OUTER surface below the outer object's rim plane is
    hidden by the outer object (a container wall is opaque), independent of small wall interpenetration.
    axis_o/centre_o in outer body frame; rim_h_o = rim height along axis_o from centre_o."""
    lo, zo, _ = rend_outer.render([T_out], roi); li, zi, _ = rend_inner.render([T_in], roi)
    x0, y0, x1, y1 = roi; K = rend_outer.K
    u, v = np.meshgrid(np.arange(x0, x1) + .5, np.arange(y0, y1) + .5)
    ray = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
    P = ray * np.nan_to_num(zi, nan=0)[..., None]                       # inner hit points, camera frame
    Ti = np.linalg.inv(T_out); Pb = P @ Ti[:3, :3].T + Ti[:3, 3] - centre_o; hb = Pb @ axis_o
    radial = Pb - hb[..., None] * axis_o; rd = ray @ Ti[:3, :3].T      # ray direction in outer frame
    inward = np.einsum('ijk,ijk->ij', rd, radial) < 0
    hidden = (li == 0) & (lo == 0) & (hb < rim_h_o) & inward & (zo > zi - eps)
    # symmetric case: the outer's INNER wall below its rim, seen from inside, is covered by an inner object lying
    # within eps behind it (the inner object sits inside the container)
    Q = ray * np.nan_to_num(zo, nan=0)[..., None]; Qb = Q @ Ti[:3, :3].T + Ti[:3, 3] - centre_o; hq = Qb @ axis_o
    rq = Qb - hq[..., None] * axis_o; outward = np.einsum('ijk,ijk->ij', rd, rq) > 0
    covered = (lo == 0) & (li == 0) & (hq < rim_h_o) & outward & (zi < zo + eps)
    has_o, has_i = (lo == 0) & ~covered, li == 0
    zo_ = np.where(has_o, zo, np.inf); zi_ = np.where(has_i & ~hidden, zi, np.inf)
    lab = np.where(np.isfinite(np.minimum(zo_, zi_)), np.where(zi_ < zo_, 1, 0), -1)
    lab = np.where(covered & ~hidden, 1, lab)
    z = np.where(lab == 1, zi, np.where(lab == 0, zo, np.nan))
    return lab, z
