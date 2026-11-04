"""ComPose tracking on the V2D ego rig: stereo point clouds + ego-motion stabilisation + mesh-based translation.

World frame = rectified-left camera at frame 0 (L0).  Steps
  1. anchor: Sim(3)-fit the canonical mesh to the frame-0 stereo points (scale from stereo -> metric mesh); pose R0, t0
  2. ComPose relative rotations per step, in L0 axes: ICP on ego-stabilised stereo clouds x hand Procrustes (WiLoR on RGB,
     rotated into rect-left axes and stabilised), blended with the learned alpha; K-frame network anchor jumps (paper A.1)
  3. translation per frame: trimmed closest-point fit of the posed metric mesh to the stereo points, R fixed
  4. variants: icp_track (6-DoF mesh ICP chained frame to frame, no network), reanchor (ComPose chain + confident 6-DoF
     mesh refinements distributed backwards)
  5. evaluation vs GT mocap (public episodes) with the kit's conventions: warp to the reference start pose, mean position /
     orientation error, ADD-AUC (1-9 cm), MPPE, SP-SR / MP-SR flags; GT time offset searched in [-3, 3]
"""
import argparse, glob, importlib.util, json, os, sys, time
import numpy as np, cv2, torch, PIL.Image, trimesh
import torchvision.transforms as tvf
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, COMPOSE, T3, work_dir, load_meta, load_ego, gt_world_T_obj, gt_mesh_path, mono_masks_dir
sys.path.insert(0, str(COMPOSE))
import icp_torch_pure as icp_mod
sys.modules['utils.icp_torch'] = icp_mod                      # ComPose's version needs pytorch3d
from utils.crop_utils import object_center_crop
from models.jointselector_temporal import JointSelectorTemporal, weighted_procrustes
from models.loss import rotation_slerp
from fit_mesh_scale import fit as sim3_fit, icp as mesh_icp, mesh_samples

DM = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); DS = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
transform = tvf.ToTensor()
KIT = ROOT / 'third_party/v2d_submission_kit/metric_code/track_3/AUC.py'


# ───────────────────────── small helpers (copied from ComPose apps/03_eval_temporal.py) ─────────────────────────
def _safe_R(R):
    R = np.asarray(R, dtype=np.float64)
    if not np.all(np.isfinite(R)): return np.eye(3)
    U, _, Vt = np.linalg.svd(R); Rp = U @ Vt
    if np.linalg.det(Rp) < 0: Vt[-1] *= -1; Rp = U @ Vt
    return Rp


def hand_bbox_mask(j2d_px, hw, pad=8):
    H, W = hw; mask = np.zeros((H, W), np.uint8)
    xs, ys = j2d_px[:, 0], j2d_px[:, 1]
    x1 = int(max(0, xs.min() - pad)); x2 = int(min(W, xs.max() + pad)); y1 = int(max(0, ys.min() - pad)); y2 = int(min(H, ys.max() + pad))
    if x2 > x1 and y2 > y1: mask[y1:y2, x1:x2] = 1
    return mask


def iou(a, b):
    return float(np.logical_and(a, b).sum()) / max(int(np.logical_or(a, b).sum()), 1)


def smooth_rotations(R_seq, window=10):
    if window <= 1 or len(R_seq) < 2: return list(R_seq)
    quats = np.stack([Rot.from_matrix(_safe_R(R)).as_quat() for R in R_seq]); half = window // 2; out = []
    for i in range(len(quats)):
        w = quats[max(0, i - half):min(len(quats), i + half + 1)].copy()
        w[(w @ w[0]) < 0] *= -1; q = w.mean(0); q /= max(np.linalg.norm(q), 1e-12); out.append(Rot.from_quat(q).as_matrix())
    return out


def T_of(R, t):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t; return T


# ───────────────────────── mesh fitting ─────────────────────────
def fit_translation(P, gR, tree, t0, iters=8, trim=0.8):
    """t minimising trimmed |P - (R g + t)| with R fixed; gR = g @ R.T, tree on gR."""
    t = t0.copy()
    for _ in range(iters):
        d, j = tree.query(P - t, k=1)
        keep = d <= np.quantile(d, trim)
        t = (P[keep] - gR[j[keep]]).mean(0)
    d, _ = tree.query(P - t, k=1)
    return t, d


def spill_fraction(g, R, t, mask_dil, K):
    """Fraction of mesh samples (posed, camera frame) projecting outside the dilated object mask: a too-large or wrongly
    placed mesh spills over the silhouette even when every observed point lies on its surface (one-sided residual is blind to it)."""
    X = g @ R.T + t; z = np.maximum(X[:, 2], 1e-6)
    u = np.round(K[0, 0] * X[:, 0] / z + K[0, 2]).astype(int); v = np.round(K[1, 1] * X[:, 1] / z + K[1, 2]).astype(int)
    H, W = mask_dil.shape; ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    inside = np.zeros(len(g), bool); inside[ok] = mask_dil[v[ok], u[ok]]
    return float(1 - inside.mean())


def coverage_fraction(g, R, t, mask, K, s=4):
    """Fraction of (raw) object-mask cells covered by the projected mesh samples (1/4-res grid): a mesh whose notch/handle
    sits over a solid part of the silhouette leaves mask cells uncovered (complement of spill_fraction)."""
    X = g @ R.T + t; z = np.maximum(X[:, 2], 1e-6)
    u = (K[0, 0] * X[:, 0] / z + K[0, 2]).astype(int) // s; v = (K[1, 1] * X[:, 1] / z + K[1, 2]).astype(int) // s
    H, W = mask.shape[0] // s, mask.shape[1] // s; ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    cov = np.zeros((H, W), bool); cov[v[ok], u[ok]] = True
    ms = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
    return float((cov & ms).sum() / max(ms.sum(), 1))


def mesh_sample_normals(path, n, seed=0):
    """Face normals of the same samples mesh_samples() returns (same seed)."""
    m = trimesh.load(path, force='mesh'); _, fid = trimesh.sample.sample_surface(m, n, seed=seed); return np.asarray(m.face_normals[fid])


def far_fraction(P, g, R, t, thr=0.008, normals=None, cam_c=None):
    """Fraction of posed mesh samples with no observed point within thr (mesh -> points direction): the solid part of a
    180-deg twin that sits over the real slot/notch, or a handle on the wrong side, has no support in the cloud. With normals,
    only camera-facing samples count (the back face never has support, for every hypothesis alike)."""
    X = g @ R.T + t; d, _ = cKDTree(P).query(X, k=1)
    if normals is not None and cam_c is not None:
        facing = np.einsum('ij,ij->i', normals @ R.T, X - cam_c) < 0
        if facing.sum() > 100: return float(np.mean(d[facing] > thr))
    return float(np.mean(d > thr))


def anchor_hypotheses(P, g, R_b, t_b, mask_dil, mask, K, E_inv, spill_w=50.0, cover_w=50.0, far_w=50.0, verbose=True, normals=None, cam_c=None):
    """Symmetric alternatives of the anchor pose (180 deg about each principal axis of the mesh: slot/handle on the other side,
    face flip of a board) refined by ICP and scored by trimmed residual + silhouette spill + uncovered mask. The one-sided
    residual cannot tell these apart; the silhouette can."""
    tree = cKDTree(g); c = g.mean(0); _, U = np.linalg.eigh(np.cov((g - c).T))
    cands = [('id', R_b, t_b)]
    for i in range(3):
        Rm = Rot.from_rotvec(np.pi * U[:, i]).as_matrix(); Rh = _safe_R(R_b @ Rm); cands.append((f'rot180-pc{i}', Rh, t_b + R_b @ c - Rh @ c))
    best = None
    for name, Rh, th in cands:
        Rh, th, _, d = refine_6dof(P, g, tree, Rh, th, iters=40)
        res = float(np.mean(np.sort(d)[:int(0.8 * len(d))]) * 1000)
        Rc, tc = (Rh, th) if E_inv is None else (E_inv[:3, :3] @ Rh, E_inv[:3, :3] @ th + E_inv[:3, 3])
        spill = spill_fraction(g, Rc, tc, mask_dil, K); cover = coverage_fraction(g, Rc, tc, mask, K); far = far_fraction(P, g, Rh, th, normals=normals, cam_c=cam_c)
        score = res + spill_w * spill + cover_w * (1 - cover) + far_w * far
        if verbose: print(f'    anchor hypothesis {name:12s}: residual {res:.2f} mm, spill {spill*100:.1f}%, uncovered {100*(1-cover):.1f}%, unsupported mesh {far*100:.1f}% -> score {score:.2f}')
        if best is None or score < best[0]: best = (score, name, Rh, th, d)
    if verbose: print(f'    anchor hypothesis chosen: {best[1]}')
    return best[2], best[3], best[4]


def fit_anchor(P, g_raw, with_scale, n_starts=60, verbose=True, mask_dil=None, K=None, spill_w=50.0, E_inv=None, mask=None, cover_w=50.0):
    """Sim(3) fit of a canonical (unitless) mesh to the stereo points. fit() starts rigid at s=1, which lets a mesh that is
    far too large swallow the partial cloud on a smooth wall; so the scale is initialised from the RMS-radius ratio on a
    small grid (the visible partial surface has a smaller radius than the full mesh -> factors >= 1) and the best trimmed
    residual wins."""
    if not with_scale:
        return sim3_fit(P, g_raw, n_starts=120, with_scale=False)
    r_obs = np.sqrt(((P - P.mean(0)) ** 2).sum(1).mean()); r_g = np.sqrt(((g_raw - g_raw.mean(0)) ** 2).sum(1).mean())
    best = None; best_score = None
    for c in (0.6, 0.8, 1.0, 1.2, 1.45, 1.75, 2.1):
        s_init = r_obs / r_g * c
        s, R, t, sc, d = sim3_fit(P, g_raw * s_init, n_starts=n_starts, with_scale=True)
        Rc, tc = (R, t) if E_inv is None else (E_inv[:3, :3] @ R, E_inv[:3, :3] @ t + E_inv[:3, 3])
        spill = spill_fraction(g_raw * (s * s_init), Rc, tc, mask_dil, K) if mask_dil is not None else 0.0
        unc = (1 - coverage_fraction(g_raw * (s * s_init), Rc, tc, mask, K)) if mask is not None else 0.0   # a too-small mesh leaves mask uncovered
        score = sc * 1000 + spill_w * spill + cover_w * unc
        if verbose: print(f'    scale init x{c:.2f} (s={s_init:.4f}) -> s={s * s_init:.4f}, trimmed residual {sc*1000:.2f} mm, silhouette spill {spill*100:.1f}%, uncovered {unc*100:.1f}% -> score {score:.2f}')
        if best is None or score < best_score: best = (s * s_init, R, t, sc, d); best_score = score
    return best


def mesh_symmetry(g, thr_mm=6.0):
    """Self-similarity of the mesh samples under rotations about each principal axis (trimmed 70 % chamfer, so a handle
    does not hide the symmetric body). Returns (axis in mesh frame, kind) with kind in {'revolution', 'twofold', None}."""
    c = g.mean(0)
    for _ in range(6):                                                  # trimmed-mean centre: the dense body, not the handle
        d = np.linalg.norm(g - c, axis=1); c = g[d <= np.quantile(d, 0.7)].mean(0)
    gc = g - c; tree = cKDTree(gc)
    w, U = np.linalg.eigh(np.cov(gc[np.linalg.norm(gc, axis=1) <= np.quantile(np.linalg.norm(gc, axis=1), 0.7)].T))
    best = (None, None, 1e9)
    for i in range(3):
        ax = U[:, i]
        sc = {}
        for deg in (90, 180):
            d, _ = tree.query(gc @ Rot.from_rotvec(np.radians(deg) * ax).as_matrix().T, k=1)
            sc[deg] = float(np.mean(np.sort(d)[:int(0.7 * len(d))]) * 1000)
        if sc[90] < thr_mm and sc[90] < best[2]: best = (ax, 'revolution', sc[90])
        elif sc[180] < thr_mm and best[1] is None and sc[180] < best[2]: best = (ax, 'twofold', sc[180])
    return best


class GraspSym:
    """Hand-grasp region as a symmetry breaker: mesh samples repeatedly within `r` of the hand points while in contact
    form the grasp region; when it drifts away from the hand, symmetric hypotheses (rotations about the symmetry axis,
    rim/bottom flip) are refined by ICP and the one that keeps the grasp region at the hand (and fits the points) wins."""
    def __init__(self, g, tree_g, axis, kind, gate_mm, thr_m=0.04, lam=1.0, r=0.025, strict=False):
        self.g, self.tree_g, self.axis, self.kind, self.gate, self.thr, self.lam, self.r = g, tree_g, axis, kind, gate_mm, thr_m, lam, r
        self.strict = strict; self.switch_log = []
        c = g.mean(0)
        for _ in range(6):
            dd = np.linalg.norm(g - c, axis=1); c = g[dd <= np.quantile(dd, 0.7)].mean(0)
        self.c = c; self.cnt = np.zeros(len(g), int); self.n_obs = 0; self.n_switch = 0
        perp = np.eye(3)[np.argmin(np.abs(axis))] if axis is not None else None
        if axis is not None:
            perp = perp - axis * (axis @ perp); perp /= np.linalg.norm(perp)
        self.perp = perp
        angs = list(range(0, 360, 30)) if kind == 'revolution' else [0, 180]
        self.hyps = [('rot', a) for a in angs] + [('flip', a) for a in angs]
    def region(self):
        if self.n_obs < 3: return None
        thr = max(3, np.quantile(self.cnt, 0.75))                           # the grasp region must be a minority of the surface
        m = self.cnt >= thr
        return m if 20 <= m.sum() <= 0.3 * len(self.cnt) else None
    def hand_dist(self, R, t, hand_tree, m):
        d, _ = hand_tree.query(self.g[m] @ R.T + t, k=1); return float(np.median(d))
    def observe(self, R, t, hand_pts):
        hand_tree = cKDTree(hand_pts)
        d, _ = hand_tree.query(self.g @ R.T + t, k=1); self.cnt[d < self.r] += 1; self.n_obs += 1
        return hand_tree
    def correct(self, P, R, t, hand_pts, f, every=10):
        """Returns (R, t, switched)."""
        if self.axis is None or hand_pts is None or len(hand_pts) < 30: return R, t, False
        hand_tree = self.observe(R, t, hand_pts); m = self.region()
        if m is None: return R, t, False
        hd0 = self.hand_dist(R, t, hand_tree, m)
        if hd0 <= self.thr and (self.strict or f % every != 0): return R, t, False   # strict: only act when the grasp region has left the hand
        d0, _ = self.tree_g.query((P - t) @ R, k=1); res0 = float(np.median(d0) * 1000); best = (res0 + self.lam * hd0 * 100, R, t, 'id')
        for kind, ang in self.hyps:
            if kind == 'rot' and ang == 0: continue
            Rm = Rot.from_rotvec(np.radians(ang) * self.axis).as_matrix()
            if kind == 'flip': Rm = Rot.from_rotvec(np.pi * self.perp).as_matrix() @ Rm
            Rh = _safe_R(R @ Rm); th = t + R @ self.c - Rh @ self.c
            Rh, th, _, d = refine_6dof(P, self.g, self.tree_g, Rh, th, iters=15)
            res = float(np.median(d) * 1000)
            if res > self.gate * 1.2: continue
            hd = self.hand_dist(Rh, th, hand_tree, m); sc = res + self.lam * hd * 100
            if self.strict:                                                   # the alternative must fit at least as well AND bring the grasp region to the hand
                if res <= res0 + 0.5 and hd < self.thr and hd < hd0 - 0.02 and sc < best[0] - 1.5: best = (sc, Rh, th, f'{kind}{ang}')
            elif sc < best[0] - 1.5 and hd < hd0 - 0.01: best = (sc, Rh, th, f'{kind}{ang}')
        if best[3] != 'id':
            self.n_switch += 1; self.switch_log.append((f, best[3], round(hd0 * 100, 1)))
            print(f'  grasp_sym switch @ frame {f}: {best[3]} (grasp region was {hd0*100:.1f} cm from the hand)', flush=True)
            return best[1], best[2], True
        return R, t, False


class PhotoFlip:
    """Appearance twin test for symmetric objects whose geometric twins (180 deg about a principal axis) fit the depth equally
    (a flat board turned over in the hand): the vertex-coloured team mesh is projected into the undistorted RGB under the pose
    and under each twin; the normalised correlation of model gray vs image gray says which face is visible. A sustained change
    of the preferred face relative to the start means the object was turned over without the ICP noticing -> switch to that twin."""
    def __init__(self, mesh_path, n, s0, wd, meta, E, win=8, margin=0.06):
        m = trimesh.load(mesh_path, force='mesh'); self.ok = m.visual.kind == 'vertex'
        if not self.ok: print('photo_flip: mesh has no vertex colours -> disabled'); return
        vc = np.asarray(m.visual.vertex_colors)[:, :3].astype(np.float32)
        P, fid = trimesh.sample.sample_surface(m, n, seed=0); self.g = np.asarray(P) * s0; self.col = vc[m.faces[fid]].mean(1).mean(1)
        c = self.g.mean(0); _, U = np.linalg.eigh(np.cov((self.g - c).T)); self.c = c; self.U = U[:, ::-1]
        self.wd, self.E, self.win, self.margin = wd, E, win, margin
        self.Ka = np.array(meta['K_a_undist'], np.float64); self.Ta = np.array(meta['T_a_from_rectL'], np.float64)
        self.hist = []; self.base = None; self.n_switch = 0; self.log = []; self._img = (None, None)
    def image(self, f):
        if self._img[0] != f:
            im = cv2.imread(str(self.wd / 'rgb_undist' / f'{f:05d}.jpg'), 0); self._img = (f, None if im is None else cv2.GaussianBlur(im, (5, 5), 0))
        return self._img[1]
    def ncc(self, f, R, t, img, s=2):
        Ei = np.linalg.inv(self.E[f]); X = self.g @ R.T + t; Xr = X @ Ei[:3, :3].T + Ei[:3, 3]; Xa = Xr @ self.Ta[:3, :3].T + self.Ta[:3, 3]
        zz = Xa[:, 2]; H, W = img.shape
        with np.errstate(divide='ignore', invalid='ignore'):
            u = self.Ka[0, 0] * Xa[:, 0] / zz + self.Ka[0, 2]; v = self.Ka[1, 1] * Xa[:, 1] / zz + self.Ka[1, 2]
        ok = (zz > 0.1) & (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
        if ok.sum() < 200: return np.nan
        idx = np.nonzero(ok)[0]; cell = (v[idx] // s).astype(np.int64) * (W // s + 1) + (u[idx] // s).astype(np.int64)
        order = np.lexsort((zz[idx], cell)); cell_s = cell[order]; first = np.ones(len(order), bool); first[1:] = cell_s[1:] != cell_s[:-1]
        sel = idx[order[first]]                                                                  # nearest sample per image cell (z-buffer)
        gi = cv2.remap(img, u[sel].astype(np.float32), v[sel].astype(np.float32), cv2.INTER_LINEAR).ravel().astype(np.float32); gm = self.col[sel]
        a_, b_ = gi - gi.mean(), gm - gm.mean(); return float((a_ * b_).sum() / (np.linalg.norm(a_) * np.linalg.norm(b_) + 1e-9))
    def twins(self, R, t):
        out = []
        for i in range(3):
            Rm = Rot.from_rotvec(np.pi * self.U[:, i]).as_matrix(); Rh = _safe_R(R @ Rm); out.append((i, Rh, t + R @ self.c - Rh @ self.c))
        return out
    def step(self, f, R, t, P, tree_g, g, gate_mm):
        if not self.ok: return R, t, False
        img = self.image(f)
        if img is None: return R, t, False
        n0 = self.ncc(f, R, t, img); tw = self.twins(R, t); nt = [self.ncc(f, Rh, th, img) for _, Rh, th in tw]
        if np.isnan(n0) or all(np.isnan(x) for x in nt): return R, t, False
        j = int(np.nanargmax(nt)); score = n0 - nt[j]                                            # > 0: the current face matches the image better than any twin
        self.hist.append(score)
        if self.base is None:                                                                    # the body frame may start 'flipped' w.r.t. the texture: only CHANGES matter
            if len(self.hist) >= 10: self.base = float(np.sign(np.median(self.hist)) or 1.0); print(f'  photo_flip baseline sign {self.base:+.0f} (median score {np.median(self.hist):+.3f})', flush=True)
            return R, t, False
        recent = np.array(self.hist[-self.win:])
        if len(recent) == self.win and np.median(recent) * self.base < -self.margin:
            Rh, th = tw[j][1], tw[j][2]; Rh, th, _, d = refine_6dof(P, g, tree_g, Rh, th, iters=30)
            if np.median(d) * 1000 <= gate_mm:
                self.n_switch += 1; self.hist = []; self.log.append((f, j)); print(f'  photo flip @ frame {f}: twin pc{j} (NCC {nt[j]:+.2f} vs current {n0:+.2f})', flush=True)
                return Rh, th, True
        return R, t, False


def refine_adaptive(P, g, tree_g, R, t, gate_mm, adaptive, iters=30):
    """Mesh ICP with trim 0.9; if its median residual fails the gate and `adaptive`, retry with trim 0.6 and judge on the
    best-60% core (occlusion by rack bars / hand leaves many off-surface points inside the mask)."""
    R2, t2, sc, d = refine_6dof(P, g, tree_g, R, t, iters=iters)
    resid = float(np.median(d) * 1000)
    if resid > gate_mm and adaptive:
        R3, t3, sc3, d3 = refine_6dof(P, g, tree_g, R, t, iters=iters + 10, trim=0.6)
        core = float(np.median(np.sort(d3)[:int(0.6 * len(d3))]) * 1000)
        if core <= gate_mm * 0.8:
            return R3, t3, core, d3, 'trim0.6'
    return R2, t2, resid, d, 'trim0.9'


def refine_6dof(P, g, tree_g, R, t, iters=30, trim=0.9, rng=None):
    rng = rng or np.random.default_rng(0)
    o = P[rng.choice(len(P), min(len(P), 4000), replace=False)]
    _, R2, t2, score, d = mesh_icp(o, tree_g, g, 1.0, R, t, iters, False, trim=trim)
    return _safe_R(R2), t2, score, d


# ───────────────────────── windows for the network ─────────────────────────
def build_window(img_path_fn, frame_order, mask_loader, j3d_s, j2d_px, pcs, K_cam):
    fd = {}
    for f in frame_order:
        m = mask_loader(f)
        if m is None: return None
        fd[f] = dict(j3d=j3d_s.get(f, np.zeros((21, 3), np.float32)), j2d_px=j2d_px.get(f, np.zeros((21, 2), np.float32)),
                     mask=m, pc=pcs.get(f, np.zeros((0, 6), np.float32)))
    f0 = None
    for fc in frame_order:
        if fd[fc]['mask'].sum() > 0:
            try:
                img_c = PIL.Image.open(img_path_fn(fc)).convert('RGB')
                crop0, _, mc0, _, _, cb = object_center_crop(img_c, fd[fc]['mask'], K_cam, scale=1.5)
                f0 = fc; break
            except Exception:
                continue
    if f0 is None: return None
    ch, cw = mc0.shape; cx1, cy1, cx2, cy2 = cb; rs = cw / (cx2 - cx1)

    def _norm(j):
        j2n = np.zeros((21, 2), np.float32)
        j2n[:, 0] = ((j[:, 0] - cx1) * rs) / cw * 2 - 1; j2n[:, 1] = ((j[:, 1] - cy1) * rs) / ch * 2 - 1
        return j2n
    imgs, masks, j2d, j3d, pcl = [], [], [], [], []
    for f in frame_order:
        if f == f0:
            img_t = transform(crop0); m_t = mc0
        else:
            img_raw = np.array(PIL.Image.open(img_path_fn(f)).convert('RGB'))
            img_t = transform(PIL.Image.fromarray(cv2.resize(img_raw[cy1:cy2, cx1:cx2], (cw, ch))))
            m_t = cv2.resize(fd[f]['mask'][cy1:cy2, cx1:cx2], (cw, ch), interpolation=cv2.INTER_NEAREST)
        imgs.append(img_t); masks.append(m_t); j2d.append(_norm(fd[f]['j2d_px'])); j3d.append(fd[f]['j3d']); pcl.append(fd[f]['pc'])
    return dict(imgs=torch.stack(imgs), masks=np.stack(masks), j2d=np.stack(j2d), j3d=np.stack(j3d), pcs=pcl, frame_order=frame_order)


@torch.no_grad()
def run_selector(windows, selector, dino, device, chunk=64):
    """windows: list of window dicts -> list of (r01, r12, r02) with R_hand (3,3) and alpha (float) each."""
    out = []
    for s in range(0, len(windows), chunk):
        ws = windows[s:s + chunk]; V = len(ws)
        imgs = torch.stack([w['imgs'] for w in ws]).to(device); _, N, C, Hi, Wi = imgs.shape
        feats = dino.get_intermediate_layers((imgs.view(V * N, C, Hi, Wi) - DM.to(device)) / DS.to(device), n=1)[0]
        feats = feats.view(V, N, Hi // 14, Wi // 14, 384)
        masks = torch.tensor(np.stack([w['masks'] for w in ws]), dtype=torch.float32, device=device)
        j2d = torch.tensor(np.stack([w['j2d'] for w in ws]), dtype=torch.float32, device=device)
        j3d = torch.tensor(np.stack([w['j3d'] for w in ws]), dtype=torch.float32, device=device)
        pa, pma = icp_mod.pad_pointclouds([pc for w in ws for pc in w['pcs']]); pa = pa.to(device); pma = pma.to(device)
        cents, _ = icp_mod.batch_centroid(pa, pma); cents = cents.view(V, N, 3)
        results, _ = selector(feats, masks, j2d, j3d, cents, pairs=[(0, 1), (1, 2), (0, 2)])
        for v in range(V):
            out.append([dict(R_hand=r['R_hand'][v].cpu().numpy(), alpha=float(r['alpha'][v].cpu())) for r in results])
    return out


@torch.no_grad()
def batched_icp(pairs, pcs, device, color_weight=0.0, chunk=128):
    """pairs: list of (fi, fj) with both in pcs -> dict (fi,fj) -> R (fj ~ R fi)."""
    out = {}
    for s in range(0, len(pairs), chunk):
        ps = pairs[s:s + chunk]
        pa, pma = icp_mod.pad_pointclouds([pcs[fi] for fi, _ in ps]); pb, pmb = icp_mod.pad_pointclouds([pcs[fj] for _, fj in ps])
        R, _ = icp_mod.batch_icp(pa.to(device), pb.to(device), pma.to(device), pmb.to(device), color_weight=color_weight)
        for k, p in enumerate(ps): out[p] = _safe_R(R[k].cpu().numpy())
    return out


def distribute_drift(R_cum, anchors_abs, fi_of_step=None):
    """R_cum: list of absolute rotations per frame; anchors_abs: {frame: R_abs}. SLERP-distribute the drift inside each chunk."""
    R_out = [R.copy() for R in R_cum]; keys = sorted(anchors_abs)
    for k in range(len(keys) - 1):
        a0, a1 = keys[k], keys[k + 1]
        R_drift = anchors_abs[a1] @ R_out[a1].T
        rv = Rot.from_matrix(_safe_R(R_drift)).as_rotvec()
        for s in range(a0 + 1, a1 + 1):
            p = (s - a0) / max(a1 - a0, 1)
            R_out[s] = _safe_R(Rot.from_rotvec(p * rv).as_matrix() @ R_out[s])
    return R_out


# ───────────────────────── evaluation (kit conventions) ─────────────────────────
def load_kit_consts():
    try:
        spec = importlib.util.spec_from_file_location('v2d_auc', KIT); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        return dict(thr=m._ADD_THRESHOLDS_M, kp=m._MPPE_KEYPOINT_DIRECTIONS * m._MPPE_KEYPOINT_RADIUS_M,
                    sp_pos=m._SPIDER_POSITION_THRESHOLD_M, sp_ori=m._SPIDER_ORIENTATION_THRESHOLD_RAD,
                    mp_pos=m._MANIPTRANS_POSITION_THRESHOLD_M, mp_ori=m._MANIPTRANS_ORIENTATION_THRESHOLD_DEG)
    except Exception as e:
        print('kit constants fallback:', e)
        return dict(thr=np.arange(0.01, 0.10, 0.01), kp=np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]) * 0.05,
                    sp_pos=0.1, sp_ori=0.5, mp_pos=0.03, mp_ori=30.0)


def evaluate(R_seq, t_seq, T_ours_from_scan, gt_T, verts, consts, d=0):
    n = len(R_seq); idx = [f for f in range(n) if 0 <= f + d < len(gt_T)]
    if len(idx) < 2: return None
    ours = np.stack([T_of(R_seq[f], t_seq[f]) @ T_ours_from_scan for f in idx])        # L0_T_scan
    gt = np.stack([gt_T[f + d] for f in idx])
    A = gt[0] @ np.linalg.inv(ours[0])                                                      # one rigid world transform from the start
    ow = np.einsum('ij,njk->nik', A, ours)
    pos = np.linalg.norm(ow[:, :3, 3] - gt[:, :3, 3], axis=1)
    ori = np.array([np.linalg.norm(Rot.from_matrix(_safe_R(ow[i, :3, :3] @ gt[i, :3, :3].T)).as_rotvec()) for i in range(len(idx))])
    pw = np.einsum('nij,vj->nvi', ow[:, :3, :3], verts) + ow[:, None, :3, 3]; pg = np.einsum('nij,vj->nvi', gt[:, :3, :3], verts) + gt[:, None, :3, 3]
    add = np.linalg.norm(pw - pg, axis=-1).mean(1)
    trap = getattr(np, 'trapezoid', None) or np.trapz
    auc = float(trap(np.array([np.mean(add < t) for t in consts['thr']]), x=np.linspace(0, 1, len(consts['thr']))))
    kw = np.einsum('nij,kj->nki', ow[:, :3, :3], consts['kp']) + ow[:, None, :3, 3]; kg = np.einsum('nij,kj->nki', gt[:, :3, :3], consts['kp']) + gt[:, None, :3, 3]
    mppe = float(np.linalg.norm(kw - kg, axis=-1).mean() * 100)
    return dict(d=d, n=len(idx), pos_cm=float(pos.mean() * 100), ori_deg=float(np.degrees(ori).mean()), auc=auc, mppe_cm=mppe,
                sp_sr=bool(pos.mean() <= consts['sp_pos'] and ori.mean() <= consts['sp_ori']),
                mp_sr=bool(pos.mean() < consts['mp_pos'] and np.degrees(ori).mean() < consts['mp_ori']),
                pos_frames=pos, ori_frames=np.degrees(ori), add_frames=add, frames=np.array(idx))


def fmt(r):
    return (f"pos {r['pos_cm']:5.2f} cm  ori {r['ori_deg']:6.2f} deg  AUC {r['auc']:.3f}  MPPE {r['mppe_cm']:5.2f} cm  "
            f"SP-SR {int(r['sp_sr'])}  MP-SR {int(r['mp_sr'])}  (d={r['d']:+d}, n={r['n']})")


# ───────────────────────── main ─────────────────────────
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ep', type=int, required=True); ap.add_argument('--obj', required=True)
    ap.add_argument('--mesh', default='team', help="'team' (Drive selected_public canonical), 'gt' (public scan, oracle) or a path")
    ap.add_argument('--ckpt', default=str(COMPOSE / 'outputs/temporal_balanced_dex1cam_hoi4d_v2/best.pth'))
    ap.add_argument('--device', default='cuda:0'); ap.add_argument('--tag', default='')
    ap.add_argument('--anchor_period', type=int, default=5); ap.add_argument('--smooth_window', type=int, default=10)
    ap.add_argument('--color_weight', type=float, default=0.0); ap.add_argument('--force_alpha', type=float, default=None)
    ap.add_argument('--iou_thresh', type=float, default=0.10); ap.add_argument('--min_obj_area', type=int, default=64)
    ap.add_argument('--no_ego', action='store_true', help='skip ego-motion stabilisation (ablation)')
    ap.add_argument('--anchor_frame', type=int, default=0)
    ap.add_argument('--photo_flip', action='store_true', help='vertex-coloured mesh vs RGB: switch to the 180-deg twin when the visible face changes (board turned over in the hand)')
    ap.add_argument('--anchor_hyp', action='store_true', help='anchor: test the 180-deg symmetric alternatives (slot/handle side, board face flip) with silhouette spill + mask coverage')
    ap.add_argument('--anchor_free_of', default='', help='anchor on the first frame where this other object (e.g. white_pot_lid) no longer touches the object mask, then track backwards to frame 0')
    ap.add_argument('--anchor_refine', type=int, default=10, help='static-phase consistency: track forward this many frames, re-refine frame 0 from the converged pose')
    ap.add_argument('--reanchor_every', type=int, default=10); ap.add_argument('--reanchor_resid_mm', type=float, default=3.0)
    ap.add_argument('--reanchor_min_pts', type=int, default=1500); ap.add_argument('--reanchor_max_deg', type=float, default=10.0)
    ap.add_argument('--n_mesh_samples', type=int, default=20000)
    ap.add_argument('--fuse_resid_mm', type=float, default=4.0); ap.add_argument('--fuse_max_deg', type=float, default=15.0)
    ap.add_argument('--fuse_min_pts', type=int, default=300)
    ap.add_argument('--icp_jump_gate_deg', type=float, default=0.0, help='icp_track/compose_meshicp: reject a mesh-ICP result that rotates more than this from the previous tracked pose in one frame (0 = off); implausible jumps = symmetry flips')
    ap.add_argument('--adaptive_trim', action='store_true', help='if the 0.9-trimmed mesh ICP is rejected by the residual gate, retry with trim 0.6 (occlusion by other objects: rack bars, hand) and judge on the best-60%% core')
    ap.add_argument('--scale_prior', action='store_true', help='use outputs/scale_prior.json (median stereo scale of confident fits of the same object) when the frame-0 fit is poor or disagrees by >5%%')
    ap.add_argument('--grasp_sym', action='store_true', help='hand-grasp region breaks object symmetry (axial slip / rim-bottom flips) in icp_track and compose_meshicp')
    ap.add_argument('--grasp_thr_m', type=float, default=0.04); ap.add_argument('--grasp_lambda', type=float, default=1.0)
    ap.add_argument('--soft_jump_gate', action='store_true', help='jump gate holds the previous rotation only while it still fits the points (symmetric flip); a jump the old rotation cannot explain is accepted as real motion')
    ap.add_argument('--grasp_strict', action='store_true', help='grasp_sym: no periodic re-check; switch only when the grasp region left the hand and the alternative fits as well and lands at the hand')
    ap.add_argument('--resid_gate_mm', type=float, default=5.0, help='reject a mesh-ICP result whose median point-to-surface residual exceeds this (mask failures)')
    ap.add_argument('--contact_px', type=int, default=300, help='hand mask overlap with the 25px-dilated object mask (mono) to call hand-object contact')
    ap.add_argument('--grasp_gate_deg', type=float, default=10.0, help='compose_grasp: max rotation the mesh ICP may move away from the hand-anchored prediction')
    ap.add_argument('--jump_gate_deg', type=float, default=15.0, help='compose_grasp: max per-frame rotation change outside contact')
    ap.add_argument('--smooth_short', type=int, default=5, help='window of the *_smooth variants (R: quaternion mean, t: mean)')
    ap.add_argument('--depth_scale', type=float, default=1.0, help='multiply the stereo metric world (points + ego translation) by this factor; stereo depth measured ~2-3%% short vs the GT scans')
    a = ap.parse_args()
    t_start = time.time(); torch.manual_seed(0); np.random.seed(0); n_softhold = [0, 0]; n_softjump = [0, 0]
    device = torch.device(a.device); ep, obj = a.ep, a.obj
    wd = work_dir(ep, obj); meta = load_meta(ep, obj); n = meta['n_frames']
    K_und = np.array(meta['K_a_undist'], np.float32); T_a = np.array(meta['T_a_from_rectL']); R_aL = T_a[:3, :3]
    out_dir = wd / f'track{a.tag}'; out_dir.mkdir(exist_ok=True)

    # ── inputs ──
    E, ego_ok = load_ego(ep, obj)
    if a.no_ego: E = np.tile(np.eye(4), (n, 1, 1))
    if a.depth_scale != 1.0:
        E = E.copy(); E[:, :3, 3] *= a.depth_scale
    RE, tE = E[:, :3, :3], E[:, :3, 3]
    pcz = np.load(wd / 'pc_cache.npz')
    pcs = {}
    for k in pcz.files:
        f = int(k[3:]); p = pcz[k].astype(np.float32)
        p[:, :3] = (p[:, :3] * a.depth_scale) @ RE[f].T + tE[f]                           # stabilised: L0 frame (metric x depth_scale)
        pcs[f] = p
    cents = {f: p[:, :3].mean(0) for f, p in pcs.items()}
    wdict = np.load(wd / 'wilor_hands.npz'); wilor_frames = {int(f): i for i, f in enumerate(wdict['frames'])}; ct = wdict['hand_pose'][:, 48:51]
    j3d_s, j2d_px = {}, {}
    for f, wi in wilor_frames.items():
        if f >= n: continue
        j = (wdict['joints_3d'][wi] + ct[wi]).astype(np.float64); c = j.mean(0)
        j3d_s[f] = ((RE[f] @ R_aL.T @ (j - c).T).T + c).astype(np.float32)               # hand shape rotated into stabilised L0 axes
        j2d_px[f] = wdict['joints_2d'][wi]
    mask_cache = {}
    def load_mask(f):
        if f not in mask_cache:
            m = cv2.imread(str(wd / 'masks_rgb' / f'{f:05d}.png'), 0)
            mask_cache[f] = None if m is None else (m > 0).astype(np.uint8)
        return mask_cache[f]
    img_path_fn = lambda f: str(wd / 'rgb_undist' / f'{f:05d}.jpg')
    # absolute hand orientation (WiLoR global orient, cam_a axes -> stabilised L0 axes) + hand-object contact from the mono masks
    R_hand_abs = {}
    for f, wi in wilor_frames.items():
        if f < n: R_hand_abs[f] = _safe_R(RE[f] @ R_aL.T @ Rot.from_rotvec(wdict['hand_pose'][wi, :3]).as_matrix())
    md_ = mono_masks_dir(ep, obj); contact = np.zeros(n, bool); contact_px = np.zeros(n, int)
    for f in range(n):
        mo_ = cv2.imread(str(md_ / f'obj_{f:05d}.png'), 0); mh_ = cv2.imread(str(md_ / f'hand_{f:05d}.png'), 0)
        if mo_ is None or mh_ is None: continue
        contact_px[f] = int(((cv2.dilate((mo_ > 0).astype(np.uint8), np.ones((25, 25), np.uint8)) > 0) & (mh_ > 0)).sum())
        contact[f] = contact_px[f] >= a.contact_px
    print(f'hand-object contact (mono masks, overlap >= {a.contact_px} px): {int(contact.sum())}/{n} frames')
    K_rect_ = np.array(meta['K_rect'], np.float64); _hand_cache = {}
    if a.grasp_sym:
        from common import load_depth, backproject
        _depth_all, _dframes, _ = load_depth(ep)
    def hand_pts(f):
        if f in _hand_cache: return _hand_cache[f]
        out = None
        if a.grasp_sym and contact[f] and f in _dframes:
            mh_ = cv2.imread(str(md_ / f'hand_{f:05d}.png'), 0)
            if mh_ is not None:
                mh_ = cv2.erode((mh_ > 0).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
                Ph, _, _ = backproject(_depth_all[_dframes.index(f)], mh_, K_rect_)
                Ph = Ph[(Ph[:, 2] > 0.15) & (Ph[:, 2] < 1.5)]
                if len(Ph) > 30:
                    if len(Ph) > 400: Ph = Ph[np.random.default_rng(f).choice(len(Ph), 400, replace=False)]
                    out = (Ph * a.depth_scale) @ RE[f].T + tE[f]
        _hand_cache[f] = out; return out
    print(f'[ep{ep} {obj}] {n} frames | pcs {len(pcs)} | hands {len(j3d_s)} | ego {"off" if a.no_ego else f"{int(ego_ok.sum())}/{n} ok"}')

    # ── mesh + anchor ──
    gt_mesh = gt_mesh_path(obj)
    if a.mesh == 'team':
        cands = sorted(glob.glob(str(ROOT / f'data/drive_results/V2D/extracted/public_meshes_20261008/selected_public/{obj}/*.obj')))
        mesh_path = cands[0]
    elif a.mesh == 'gt':
        mesh_path = str(gt_mesh)
    else:
        mesh_path = a.mesh
    g_raw = mesh_samples(mesh_path, a.n_mesh_samples); n_raw = mesh_sample_normals(mesh_path, a.n_mesh_samples) if a.anchor_hyp else None
    K_rect = np.array(meta['K_rect'], np.float64)
    if a.anchor_free_of:
        # lid-on pot: at frame 0 the lid closes the pot and the Sim(3) anchor fit cannot tell upright from upside-down -> anchor where the lid is off
        md_o = mono_masks_dir(ep, a.anchor_free_of); md_s = mono_masks_dir(ep, obj); contact_frac = np.full(n, np.nan)
        for f in range(n):
            ms_ = cv2.imread(str(md_s / f'obj_{f:05d}.png'), 0); mo_ = cv2.imread(str(md_o / f'obj_{f:05d}.png'), 0)
            if ms_ is None or mo_ is None or not mo_.any(): continue
            mo_ = mo_ > 0; rest = ((ms_ > 0) & ~mo_).astype(np.uint8)
            if rest.sum() < 200: continue
            hull = np.zeros_like(rest); cv2.fillConvexPoly(hull, cv2.convexHull(np.argwhere(rest > 0)[:, ::-1].astype(np.int32)), 1)
            contact_frac[f] = (hull > 0)[mo_].mean()                                             # lid inside the pot's convex hull = still on it
        free = np.array([(f in pcs and len(pcs[f]) >= 1000 and not np.isnan(contact_frac[f]) and contact_frac[f] < 0.2) for f in range(n)])
        cand = [f for f in range(n - 10) if free[f:f + 10].all()]
        if cand:
            a.anchor_frame = min(cand[0] + 5, n - 1)
            a._anchor_cands = [f for f in (cand[0] + 5, cand[0] + 20, cand[0] + 40, cand[0] + 60) if f < n]   # the best-fitting frame of the free run wins
            print(f'anchor_free_of {a.anchor_free_of}: contact fraction at frame 0 {contact_frac[0]:.2f}; first free run at {cand[0]} -> anchor candidates {a._anchor_cands} (backward track to 0)')
        else:
            print(f'anchor_free_of {a.anchor_free_of}: never free (min contact {np.nanmin(contact_frac):.2f}); anchor stays at frame {a.anchor_frame}')
    if a.anchor_frame not in pcs or len(pcs[a.anchor_frame]) < 500:
        good = [f for f in range(n) if f in pcs and len(pcs[f]) >= 500]
        if not good: raise RuntimeError('no frame with enough object points')
        print(f'anchor frame {a.anchor_frame} has too few points -> using frame {good[0]} (poses before it hold the anchor pose)')
        a.anchor_frame = good[0]
    if len(getattr(a, '_anchor_cands', [])) > 1:
        fits = []
        for fr in a._anchor_cands:
            if fr not in pcs or len(pcs[fr]) < 500: continue
            P_ = pcs[fr][:, :3].astype(np.float64); m_ = cv2.imread(str(mono_masks_dir(ep, obj) / f'obj_{fr:05d}.png'), 0)
            mdil_ = (cv2.dilate((m_ > 0).astype(np.uint8), np.ones((21, 21), np.uint8)) > 0) if m_ is not None else None
            r_ = fit_anchor(P_, g_raw, with_scale=(a.mesh != 'gt'), mask_dil=mdil_, K=K_rect, E_inv=np.linalg.inv(E[fr]), verbose=False,
                            mask=((m_ > 0) if (a.anchor_hyp and m_ is not None) else None))
            fits.append((r_[3], fr)); print(f'  anchor candidate frame {fr}: trimmed residual {r_[3]*1000:.2f} mm, scale {r_[0]:.4f}', flush=True)
        if fits: a.anchor_frame = min(fits)[1]; print(f'  -> anchor frame {a.anchor_frame}')
    P0 = pcs[a.anchor_frame][:, :3].astype(np.float64)
    m_anchor = cv2.imread(str(mono_masks_dir(ep, obj) / f'obj_{a.anchor_frame:05d}.png'), 0)
    mask_dil = (cv2.dilate((m_anchor > 0).astype(np.uint8), np.ones((21, 21), np.uint8)) > 0) if m_anchor is not None else None
    s0, R0, t0, sc0, d0 = fit_anchor(P0, g_raw, with_scale=(a.mesh != 'gt'), mask_dil=mask_dil, K=K_rect, E_inv=np.linalg.inv(E[a.anchor_frame]),
                                     mask=((m_anchor > 0) if (a.anchor_hyp and m_anchor is not None) else None))
    print(f'anchor @ frame {a.anchor_frame} ({a.mesh} mesh): scale {s0:.4f}, trimmed residual {sc0*1000:.2f} mm, median {np.median(d0)*1000:.2f} mm, p90 {np.percentile(d0, 90)*1000:.2f} mm')
    if a.anchor_hyp and m_anchor is not None:
        R0, t0, d0 = anchor_hypotheses(P0, g_raw * s0, R0, t0, mask_dil, m_anchor > 0, K_rect, np.linalg.inv(E[a.anchor_frame]), normals=n_raw, cam_c=E[a.anchor_frame][:3, 3])
    sp_path = ROOT / 'outputs/scale_prior.json'
    if a.scale_prior and a.mesh != 'gt' and sp_path.exists() and obj in json.load(open(sp_path)):
        s_pr = json.load(open(sp_path))[obj]
        if abs(s0 / s_pr - 1) > 0.05 or sc0 * 1000 > 2.5:
            _, R0p, t0p, sc0p, d0p = sim3_fit(P0, g_raw * s_pr, n_starts=120, with_scale=False)
            print(f'scale prior {s_pr:.4f} (fit {s0:.4f}, {abs(s0/s_pr-1)*100:.1f}% off): rigid refit residual {sc0p*1000:.2f} mm -> using prior')
            s0, R0, t0, sc0, d0 = s_pr, R0p, t0p, sc0p, d0p
            if a.anchor_hyp and m_anchor is not None:
                R0, t0, d0 = anchor_hypotheses(P0, g_raw * s0, R0, t0, mask_dil, m_anchor > 0, K_rect, np.linalg.inv(E[a.anchor_frame]), normals=n_raw, cam_c=E[a.anchor_frame][:3, 3])
    g = g_raw * s0                                                                           # metric canonical mesh (body frame)
    tree_g = cKDTree(g)
    if a.anchor_refine > 0:
        # the multi-start Sim(3) fit can stop in a local optimum a few degrees off (symmetric body, thin handle); the scorer
        # aligns everything on the initial pose, so that error would become a constant offset over the whole episode.
        Rr, tr = R0.copy(), t0.copy()
        for f in range(a.anchor_frame + 1, min(a.anchor_frame + a.anchor_refine + 1, n)):
            if f in pcs and len(pcs[f]) >= 200:
                Rr, tr, _, _ = refine_6dof(pcs[f][:, :3].astype(np.float64), g, tree_g, Rr, tr)
        R0b, t0b, _, d0b = refine_6dof(P0, g, tree_g, Rr, tr, iters=60)
        dRa = np.degrees(np.linalg.norm(Rot.from_matrix(_safe_R(R0b @ R0.T)).as_rotvec())); dta = np.linalg.norm(t0b - t0) * 100
        print(f'anchor refine (fwd {a.anchor_refine} frames + back to frame {a.anchor_frame}): moved {dRa:.2f} deg / {dta:.2f} cm; residual median {np.median(d0)*1000:.2f} -> {np.median(d0b)*1000:.2f} mm')
        if np.median(d0b) <= np.median(d0) * 1.05:
            R0, t0, d0 = R0b, t0b, d0b
    sym_axis, sym_kind, sym_sc = mesh_symmetry(g)
    print(f'mesh symmetry: {sym_kind} (self-chamfer {sym_sc:.2f} mm)' if sym_kind else 'mesh symmetry: none detected')
    if a.anchor_frame > 0:
        # backward mesh-ICP pass anchor -> 0 so that frame 0 (which the scorer aligns on) carries the anchor's basin
        Rb, tb = R0.copy(), t0.copy(); nb_ok = 0; nb_rej = 0
        for f in range(a.anchor_frame - 1, -1, -1):
            tinit = tb + ((cents[f] - cents[f + 1]) if (f in cents and f + 1 in cents) else 0)
            if f in pcs and len(pcs[f]) >= 200:
                R2, t2, resid2, d2, _ = refine_adaptive(pcs[f][:, :3].astype(np.float64), g, tree_g, Rb, tinit, a.resid_gate_mm, a.adaptive_trim)
                jump = np.degrees(np.linalg.norm(Rot.from_matrix(_safe_R(R2 @ Rb.T)).as_rotvec()))
                if resid2 <= a.resid_gate_mm and (a.icp_jump_gate_deg <= 0 or jump <= a.icp_jump_gate_deg): Rb, tb = R2, t2; nb_ok += 1
                else: tb = tinit; nb_rej += 1
            else: tb = tinit
        dRb = np.degrees(np.linalg.norm(Rot.from_matrix(_safe_R(Rb @ R0.T)).as_rotvec())); dtb = np.linalg.norm(tb - t0) * 100
        print(f'backward track {a.anchor_frame} -> 0: {nb_ok} ICP frames, {nb_rej} rejected; frame-0 pose {dRb:.1f} deg / {dtb:.1f} cm from the anchor pose')
        R0, t0 = Rb, tb
        if 0 in pcs: d0, _ = tree_g.query((pcs[0][:, :3].astype(np.float64) - t0) @ R0, k=1)
    pf_it = PhotoFlip(mesh_path, a.n_mesh_samples, s0, wd, meta, E) if a.photo_flip else None
    pf_cm = PhotoFlip(mesh_path, a.n_mesh_samples, s0, wd, meta, E) if a.photo_flip else None
    gs_it = GraspSym(g, tree_g, sym_axis, sym_kind, a.resid_gate_mm, a.grasp_thr_m, a.grasp_lambda, strict=a.grasp_strict) if a.grasp_sym else None
    gs_cm = GraspSym(g, tree_g, sym_axis, sym_kind, a.resid_gate_mm, a.grasp_thr_m, a.grasp_lambda, strict=a.grasp_strict) if a.grasp_sym else None
    m_tri = trimesh.load(mesh_path, force='mesh'); m_tri.vertices = m_tri.vertices * s0; m_tri.export(out_dir / 'mesh_metric.obj')
    # body registration to the GT scan (for evaluation only): ours ~ R_b scan + t_b
    has_gt = gt_mesh is not None and meta['split'] == 'public'
    if has_gt:
        gt_tri = trimesh.load(str(gt_mesh), force='mesh'); gt_verts = np.asarray(gt_tri.vertices, np.float64)
        gt_verts_eval = gt_verts[np.random.default_rng(0).choice(len(gt_verts), min(2000, len(gt_verts)), replace=False)]
    if not has_gt:
        T_ours_from_scan = np.eye(4); s_b = float('nan'); print('no public GT scan for this object: evaluation skipped')
    elif a.mesh == 'gt':
        T_ours_from_scan = np.eye(4); s_b = 1.0
    else:
        g_scan = mesh_samples(str(gt_mesh), 30000)
        s_b, R_b, t_b, sc_b, _ = sim3_fit(g, g_scan, n_starts=120, with_scale=True)
        _, R_b, t_b, sc_b2, _ = sim3_fit(g, g_scan, n_starts=120, with_scale=False)
        T_ours_from_scan = T_of(R_b, t_b)
        print(f'body registration ours<-scan: scale {s_b:.4f} (1 = stereo-metric mesh matches the scan), rigid residual {sc_b2*1000:.2f} mm')

    # ── per-step ICP (stabilised clouds) and uniform hand Procrustes ──
    step_pairs = [(f, f + 1) for f in range(n - 1) if f in pcs and f + 1 in pcs]
    R_icp = batched_icp(step_pairs, pcs, device, a.color_weight)
    R_hand_u = {}
    for f in range(n - 1):
        if f in j3d_s and f + 1 in j3d_s:
            ji = torch.tensor(j3d_s[f])[None]; jj = torch.tensor(j3d_s[f + 1])[None]
            R_hand_u[f] = _safe_R(weighted_procrustes(ji, jj, torch.ones(1, 21) / 21)[0].numpy())
    print(f'ICP steps {len(R_icp)}/{n-1}, hand steps {len(R_hand_u)}/{n-1}')

    # ── network windows ──
    print('loading DINOv2 + ComPose selector ...')
    dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').eval().to(device)
    selector = JointSelectorTemporal(); ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    selector.load_state_dict(ck['model'] if 'model' in ck else ck); selector.eval().to(device)
    win_info = {}; windows = []
    for fi in range(n - 1):
        fj = fi + 1
        mi, mj = load_mask(fi), load_mask(fj)
        if mi is None or mj is None or fi not in wilor_frames or fj not in wilor_frames: continue
        if mi.sum() < a.min_obj_area or mj.sum() < a.min_obj_area: continue
        hw = mi.shape
        io = max(iou(mi, hand_bbox_mask(j2d_px[fi], hw)), iou(mj, hand_bbox_mask(j2d_px[fj], hw)))
        if fj + 1 < n and (fj + 1) in wilor_frames and load_mask(fj + 1) is not None:
            order, pair_idx = (fi, fj, fj + 1), 0
        elif fi - 1 >= 0 and (fi - 1) in wilor_frames and load_mask(fi - 1) is not None:
            order, pair_idx = (fi - 1, fi, fj), 1
        else:
            order, pair_idx = (fi, fj, fj), 0
        w = build_window(img_path_fn, order, load_mask, j3d_s, j2d_px, pcs, K_und)
        if w is None: continue
        win_info[fi] = dict(idx=len(windows), pair_idx=pair_idx, rot_fb=io < a.iou_thresh, iou=io); windows.append(w)
    anc_frames = list(range(0, n, a.anchor_period))
    if anc_frames[-1] != n - 1: anc_frames.append(n - 1)
    anc_windows, anc_keys = [], []
    for k in range(len(anc_frames) - 1):
        a0, a1 = anc_frames[k], anc_frames[k + 1]; mid = a0 + 1 if a0 + 1 < a1 else a0
        if not all(f in wilor_frames and load_mask(f) is not None for f in (a0, mid, a1)): continue
        w = build_window(img_path_fn, (a0, mid, a1), load_mask, j3d_s, j2d_px, pcs, K_und)
        if w is None: continue
        anc_windows.append(w); anc_keys.append((a0, a1))
    ious = np.array([v['iou'] for v in win_info.values()]); print(f'hand-object bbox IoU percentiles 10/25/50/75/90: {np.percentile(ious, [10,25,50,75,90]).round(3)}')
    print(f'windows: {len(windows)} step windows ({sum(v["rot_fb"] for v in win_info.values())} low-IoU -> ICP only), {len(anc_windows)} anchor windows (K={a.anchor_period})')
    res_step = run_selector(windows, selector, dino, device)
    res_anc = run_selector(anc_windows, selector, dino, device)
    R_icp_anc = batched_icp([p for p in anc_keys if p[0] in pcs and p[1] in pcs], pcs, device, a.color_weight)

    # ── relative rotation per step ──
    R_step = []; src = []; alphas = []
    for fi in range(n - 1):
        wi = win_info.get(fi); R = np.eye(3); how = 'identity'
        if wi is not None:
            r = res_step[wi['idx']][wi['pair_idx']]; al = r['alpha'] if a.force_alpha is None else a.force_alpha
            if (fi, fi + 1) in R_icp and not wi['rot_fb']:
                R = _safe_R(rotation_slerp(torch.tensor(R_icp[(fi, fi + 1)])[None].float(), torch.tensor(r['R_hand'])[None].float(), torch.tensor([al]))[0].numpy())
                how = 'blend'; alphas.append(al)
            elif (fi, fi + 1) in R_icp:
                R = R_icp[(fi, fi + 1)]; how = 'icp(lowiou)'
            else:
                R = _safe_R(r['R_hand']); how = 'hand_net'
        elif (fi, fi + 1) in R_icp:
            R = R_icp[(fi, fi + 1)]; how = 'icp'
        elif fi in R_hand_u:
            R = R_hand_u[fi]; how = 'hand_uniform'
        R_step.append(R); src.append(how)
    from collections import Counter
    print('step sources:', dict(Counter(src)), f'| alpha mean {np.mean(alphas) if alphas else float("nan"):.3f}')

    # ── cumulative rotation chains ──
    R_cum = [R0.copy()]
    for s in range(n - 1): R_cum.append(_safe_R(R_step[s] @ R_cum[-1]))
    anchors_abs = {0: R0.copy()}; cur = R0.copy()
    for k, (a0, a1) in enumerate(anc_keys):
        r02 = res_anc[k][2]; al = r02['alpha'] if a.force_alpha is None else a.force_alpha
        if (a0, a1) in R_icp_anc:
            jump = _safe_R(rotation_slerp(torch.tensor(R_icp_anc[(a0, a1)])[None].float(), torch.tensor(r02['R_hand'])[None].float(), torch.tensor([al]))[0].numpy())
        else:
            jump = _safe_R(r02['R_hand'])
        cur = _safe_R(jump @ cur); anchors_abs[a1] = cur
    R_cum_anc = distribute_drift(R_cum, anchors_abs)

    # ── translation: mesh fit with R fixed (+ centroid baseline) ──
    def fit_t_chain(R_list):
        t_list = []; resid = []; t_prev = t0.copy(); f_prev = None
        for f in range(n):
            tinit = t_prev + ((cents[f] - cents[f_prev]) if (f in cents and f_prev in cents) else 0)
            if f in pcs and len(pcs[f]) >= 200:
                gR = g @ R_list[f].T; tree = cKDTree(gR)
                t, d = fit_translation(pcs[f][:, :3].astype(np.float64), gR, tree, tinit)
                resid.append(float(np.median(d) * 1000))
            else:
                t = tinit; resid.append(np.nan)
            t_list.append(t); t_prev = t; f_prev = f if f in cents else f_prev
        return np.stack(t_list), np.array(resid)
    t_mesh, resid_mesh = fit_t_chain(R_cum)
    t_mesh_anc, resid_mesh_anc = fit_t_chain(R_cum_anc)
    # centroid translation (original ComPose): offset so that frame 0 matches the anchor
    off = t0 - cents.get(0, t0); t_cent = np.stack([cents[f] + off if f in cents else (t_mesh[f]) for f in range(n)])

    # ── icp_track: chained 6-DoF mesh ICP (no network) ──
    R_it = [R0.copy()]; t_it = [t0.copy()]; resid_it = [float(np.median(d0) * 1000)]
    for f in range(1, n):
        Rp, tp = R_it[-1], t_it[-1]
        tinit = tp + ((cents[f] - cents[f - 1]) if (f in cents and f - 1 in cents) else 0)
        if f in pcs and len(pcs[f]) >= 200:
            R2, t2, resid2, d, _how = refine_adaptive(pcs[f][:, :3].astype(np.float64), g, tree_g, Rp, tinit, a.resid_gate_mm, a.adaptive_trim)
            jump = np.degrees(np.linalg.norm(Rot.from_matrix(_safe_R(R2 @ Rp.T)).as_rotvec()))
            if a.soft_jump_gate and a.icp_jump_gate_deg > 0 and jump > a.icp_jump_gate_deg and resid2 <= a.resid_gate_mm:
                # soft gate: if the previous rotation still explains the points (symmetric ambiguity) continuity wins, otherwise the jump is real motion
                gR = g @ Rp.T; tree = cKDTree(gR); t_h, d_h = fit_translation(pcs[f][:, :3].astype(np.float64), gR, tree, tinit)
                if np.median(d_h) * 1000 <= a.resid_gate_mm: R2, t2, resid2 = Rp, t_h, float(np.median(d_h) * 1000); n_softhold[0] += 1
                else: n_softjump[0] += 1
                jump = 0.0
            if resid2 <= a.resid_gate_mm and (a.icp_jump_gate_deg <= 0 or jump <= a.icp_jump_gate_deg):
                if gs_it is not None and contact[f]:
                    R2, t2, _sw = gs_it.correct(pcs[f][:, :3].astype(np.float64), R2, t2, hand_pts(f), f)
                if pf_it is not None: R2, t2, _ = pf_it.step(f, R2, t2, pcs[f][:, :3].astype(np.float64), tree_g, g, a.resid_gate_mm)
                R_it.append(R2); t_it.append(t2); resid_it.append(resid2)
            else:                                                                  # garbage fit (mask failure): keep the prediction
                gR = g @ Rp.T; tree = cKDTree(gR); t2, d2 = fit_translation(pcs[f][:, :3].astype(np.float64), gR, tree, tinit)
                R_it.append(Rp); t_it.append(t2 if np.median(d2) * 1000 <= a.resid_gate_mm else tinit); resid_it.append(float(np.median(d) * 1000))
        else:
            R_it.append(Rp); t_it.append(tinit); resid_it.append(np.nan)

    # ── reanchor: ComPose chain + confident 6-DoF refinements, drift distributed backwards ──
    R_re = [R0.copy()]; last_anchor = 0; n_re = 0
    for f in range(1, n):
        R_re.append(_safe_R(R_step[f - 1] @ R_re[-1]))
        if f % a.reanchor_every == 0 and f in pcs and len(pcs[f]) >= a.reanchor_min_pts:
            gR = g @ R_re[f].T; tree = cKDTree(gR)
            tinit = t_mesh[f]
            t_f, d = fit_translation(pcs[f][:, :3].astype(np.float64), gR, tree, tinit)
            R2, t2, sc, d2 = refine_6dof(pcs[f][:, :3].astype(np.float64), g, tree_g, R_re[f], t_f)
            dR = np.degrees(np.linalg.norm(Rot.from_matrix(_safe_R(R2 @ R_re[f].T)).as_rotvec()))
            if np.median(d2) * 1000 <= a.reanchor_resid_mm and dR <= a.reanchor_max_deg and np.median(d2) < np.median(d):
                R_re = distribute_drift(R_re, {last_anchor: R_re[last_anchor], f: R2}); last_anchor = f; n_re += 1
    t_re, resid_re = fit_t_chain(R_re)
    print(f'reanchor: {n_re} accepted refinements')

    # ── fused: ComPose-predicted, stereo/mesh-corrected (predict -> correct) ──
    R_fu = [R0.copy()]; t_fu = [t0.copy()]; resid_fu = [float(np.median(d0) * 1000)]; n_corr = 0; fu_src = []
    for f in range(1, n):
        R_pred = _safe_R(R_step[f - 1] @ R_fu[-1]); t_pred = t_fu[-1] + ((cents[f] - cents[f - 1]) if (f in cents and f - 1 in cents) else 0)
        if f in pcs and len(pcs[f]) >= 200:
            P = pcs[f][:, :3].astype(np.float64)
            R2, t2, sc, d = refine_6dof(P, g, tree_g, R_pred, t_pred)
            dR = np.degrees(np.linalg.norm(Rot.from_matrix(_safe_R(R2 @ R_pred.T)).as_rotvec()))
            if np.median(d) * 1000 <= a.fuse_resid_mm and dR <= a.fuse_max_deg and len(P) >= a.fuse_min_pts:
                R_fu.append(R2); t_fu.append(t2); resid_fu.append(float(np.median(d) * 1000)); n_corr += 1; fu_src.append('corrected'); continue
            gR = g @ R_pred.T; tree = cKDTree(gR); t_f, d = fit_translation(P, gR, tree, t_pred)
            R_fu.append(R_pred); t_fu.append(t_f); resid_fu.append(float(np.median(d) * 1000)); fu_src.append('predicted(t fit)')
        else:
            R_fu.append(R_pred); t_fu.append(t_pred); resid_fu.append(np.nan); fu_src.append('predicted')
    print(f'fused: {dict(Counter(fu_src))}')

    # ── compose_meshicp: same 2-way blend, ICP cue := absolute mesh-ICP pose relative to the tracked pose
    #    R(f) = SLERP(R_icp, R_hand, alpha) R(f-1) with R_icp = R_mesh(f) R(f-1)^T  <=>  SLERP(R_mesh(f), R_hand R(f-1), alpha)
    #    alpha -> 0 snaps to the absolute mesh pose (no drift), alpha -> 1 propagates with the hand; no IoU gate.
    R_cm = [R0.copy()]; t_cm = [t0.copy()]; resid_cm = [float(np.median(d0) * 1000)]; cm_src = []
    for f in range(1, n):
        Rp, tp = R_cm[-1], t_cm[-1]
        wi = win_info.get(f - 1)
        in_contact = bool(contact[f]) and bool(contact[f - 1])                     # hand cue only while the hand is on the object
        if wi is not None and in_contact:
            r = res_step[wi['idx']][wi['pair_idx']]; R_hand_f = _safe_R(r['R_hand']); al = r['alpha'] if a.force_alpha is None else a.force_alpha
        else:
            R_hand_f = np.eye(3); al = 0.0
        R_pred = _safe_R(R_hand_f @ Rp)
        t_pred = tp + ((cents[f] - cents[f - 1]) if (f in cents and f - 1 in cents) else 0)
        if f in pcs and len(pcs[f]) >= 200:
            P = pcs[f][:, :3].astype(np.float64)
            R_m, t_m, resid_m, d, _how = refine_adaptive(P, g, tree_g, R_pred, t_pred, a.resid_gate_mm, a.adaptive_trim)   # mesh ICP from the (hand-)predicted pose
            jump = np.degrees(np.linalg.norm(Rot.from_matrix(_safe_R(R_m @ Rp.T)).as_rotvec()))
            if a.soft_jump_gate and a.icp_jump_gate_deg > 0 and jump > a.icp_jump_gate_deg and resid_m <= a.resid_gate_mm:
                gR = g @ Rp.T; tree = cKDTree(gR); t_h, d_h = fit_translation(P, gR, tree, t_pred)
                if np.median(d_h) * 1000 <= a.resid_gate_mm:                      # previous rotation still fits: hold it (symmetric ambiguity)
                    R_pred = Rp; al = 0.0; R_m, t_m, resid_m = Rp, t_h, float(np.median(d_h) * 1000); n_softhold[1] += 1
                else: n_softjump[1] += 1                                          # it does not: real fast motion, accept the ICP pose
                jump = 0.0
            elif a.icp_jump_gate_deg > 0 and jump > a.icp_jump_gate_deg:         # implausible jump (symmetry flip): hold rotation
                R_pred = Rp; al = 0.0
                R_m, t_m, resid_m, d, _how = refine_adaptive(P, g, tree_g, Rp, t_pred, a.resid_gate_mm, a.adaptive_trim)
                jump = np.degrees(np.linalg.norm(Rot.from_matrix(_safe_R(R_m @ Rp.T)).as_rotvec()))
            if resid_m <= a.resid_gate_mm and (a.icp_jump_gate_deg <= 0 or jump <= a.icp_jump_gate_deg):
                R_f = _safe_R(rotation_slerp(torch.tensor(R_m)[None].float(), torch.tensor(R_pred)[None].float(), torch.tensor([al]))[0].numpy())
                gR = g @ R_f.T; tree = cKDTree(gR); t_f, d2 = fit_translation(P, gR, tree, t_m)
                if gs_cm is not None and contact[f]:
                    R_f, t_f, _sw = gs_cm.correct(P, R_f, t_f, hand_pts(f), f)
                if pf_cm is not None: R_f, t_f, _ = pf_cm.step(f, R_f, t_f, P, tree_g, g, a.resid_gate_mm)
                R_cm.append(R_f); t_cm.append(t_f); resid_cm.append(float(np.median(d2) * 1000)); cm_src.append('blend' if al > 0 else 'icp')
            else:
                gR = g @ R_pred.T; tree = cKDTree(gR); t_f, d2 = fit_translation(P, gR, tree, t_pred)
                R_cm.append(R_pred); t_cm.append(t_f if np.median(d2) * 1000 <= a.resid_gate_mm else t_pred); resid_cm.append(float(np.median(d) * 1000)); cm_src.append('rejected(resid)')
        else:
            R_cm.append(R_pred); t_cm.append(t_pred); resid_cm.append(np.nan); cm_src.append('hand')
    if pf_it is not None: print(f'photo_flip switches: icp_track {pf_it.n_switch} {pf_it.log}, compose_meshicp {pf_cm.n_switch} {pf_cm.log}')
    if a.soft_jump_gate: print(f'soft jump gate: icp_track held {n_softhold[0]} / accepted {n_softjump[0]} jumps; compose_meshicp held {n_softhold[1]} / accepted {n_softjump[1]}')
    print(f'compose_meshicp: {dict(Counter(cm_src))}' + (f' | grasp_sym switches: icp_track {gs_it.n_switch}, compose_meshicp {gs_cm.n_switch}, region {int((gs_cm.cnt >= 3).sum())} samples' if gs_cm is not None else ''))

    # ── compose_grasp: hand-anchored prediction (absolute WiLoR orientation x grasp transform fixed at contact onset),
    #    mesh ICP accepted only within a rotation gate (kills symmetry flips when only a box-like part is visible)
    R_gr = [R0.copy()]; t_gr = [t0.copy()]; gr_src = []; offset = None
    for f in range(1, n):
        Rp, tp = R_gr[-1], t_gr[-1]
        use_hand = bool(contact[f]) and f in R_hand_abs
        if use_hand:
            if offset is None:
                fr = f - 1 if (f - 1) in R_hand_abs else f
                offset = _safe_R(R_hand_abs[fr].T @ Rp)                                  # grasp transform from the last tracked pose
            R_pred = _safe_R(R_hand_abs[f] @ offset); gate = a.grasp_gate_deg
        else:
            offset = None; R_pred = Rp; gate = a.jump_gate_deg
        t_pred = tp + ((cents[f] - cents[f - 1]) if (f in cents and f - 1 in cents) else 0)
        if f in pcs and len(pcs[f]) >= 200:
            P = pcs[f][:, :3].astype(np.float64)
            R_m, t_m, _, d = refine_6dof(P, g, tree_g, R_pred, t_pred)
            dR = np.degrees(np.linalg.norm(Rot.from_matrix(_safe_R(R_m @ R_pred.T)).as_rotvec()))
            if dR <= gate and np.median(d) * 1000 <= a.fuse_resid_mm * 1.5:
                R_f, t_f = R_m, t_m; gr_src.append('icp' + ('(contact)' if use_hand else ''))
                if use_hand: offset = _safe_R(R_hand_abs[f].T @ R_f)                      # follow slow in-hand slip
            else:
                gR = g @ R_pred.T; tree = cKDTree(gR); t_f, _ = fit_translation(P, gR, tree, t_pred); R_f = R_pred
                gr_src.append('hand-pred' if use_hand else 'prev-pred')
        else:
            R_f, t_f = R_pred, t_pred; gr_src.append('no-pts')
        R_gr.append(R_f); t_gr.append(t_f)
    print(f'compose_grasp: {dict(Counter(gr_src))}')

    def smooth_poses(R_list, t_arr, window):
        half = window // 2
        return smooth_rotations(R_list, window), np.stack([t_arr[max(0, i - half):i + half + 1].mean(0) for i in range(len(t_arr))])
    R_cm_s, t_cm_s = smooth_poses(R_cm, np.stack(t_cm), a.smooth_short)
    R_gr_s, t_gr_s = smooth_poses(R_gr, np.stack(t_gr), a.smooth_short)

    # smoothing (rotation only, as in ComPose)
    R_cum_sm = smooth_rotations(R_cum, a.smooth_window); R_cum_anc_sm = smooth_rotations(R_cum_anc, a.smooth_window)
    t_sm_anc, _ = fit_t_chain(R_cum_anc_sm)

    variants = {
        'compose_chain(t=mesh)': (R_cum, t_mesh), 'compose_chain(t=centroid)': (R_cum, t_cent),
        'compose_anchorK(t=mesh)': (R_cum_anc, t_mesh_anc), 'compose_anchorK_smooth(t=mesh)': (R_cum_anc_sm, t_sm_anc),
        'compose_reanchor(t=mesh)': (R_re, t_re), 'icp_track_6dof': (R_it, np.stack(t_it)),
        'fused_predict_correct': (R_fu, np.stack(t_fu)), 'compose_meshicp': (R_cm, np.stack(t_cm)),
        'compose_meshicp_smooth': (R_cm_s, t_cm_s), 'compose_grasp': (R_gr, np.stack(t_gr)), 'compose_grasp_smooth': (R_gr_s, t_gr_s),
    }
    np.savez(out_dir / 'poses.npz', E=E, R0=R0, t0=t0, scale=s0, T_ours_from_scan=T_ours_from_scan,
             **{k.replace('(', '_').replace(')', '').replace('=', ''): np.stack([T_of(R, t) for R, t in zip(Rs, ts)]) for k, (Rs, ts) in variants.items()},
             resid_mesh=resid_mesh, resid_it=np.array(resid_it), resid_re=resid_re, resid_fu=np.array(resid_fu), resid_cm=np.array(resid_cm), contact=contact, contact_px=contact_px, n_pts=np.array([len(pcs[f]) if f in pcs else 0 for f in range(n)]),
             alpha_steps=np.array([res_step[win_info[f]['idx']][win_info[f]['pair_idx']]['alpha'] if f in win_info else np.nan for f in range(n - 1)]))

    # ── evaluation ──
    results = {}
    if has_gt:
        consts = load_kit_consts(); gt_T, gt_vis = gt_world_T_obj(ep, obj)
        print(f'\nGT: {len(gt_T)} rows, visible {int(gt_vis.sum())} | residual to mesh (median mm): chain {np.nanmedian(resid_mesh):.2f}, icp_track {np.nanmedian(resid_it):.2f}, reanchor {np.nanmedian(resid_re):.2f}')
        print(f'{"variant":34s} {"d=0":>70s}   | best offset')
        for name, (Rs, ts) in variants.items():
            r0 = evaluate(Rs, ts, T_ours_from_scan, gt_T, gt_verts_eval, consts, 0)
            best = min((evaluate(Rs, ts, T_ours_from_scan, gt_T, gt_verts_eval, consts, d) for d in range(-3, 4)), key=lambda r: r['pos_cm'] + r['ori_deg'] / 10)
            print(f'{name:34s} {fmt(r0)}   | {fmt(best)}')
            results[name] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in best.items()}
            results[name]['d0'] = {k: v for k, v in r0.items() if not isinstance(v, np.ndarray)}
        # error-over-time diagnostics
        for name in ('compose_anchorK_smooth(t=mesh)', 'icp_track_6dof', 'compose_meshicp', 'compose_grasp'):
            r = results[name]; of = np.array(r['ori_frames']); pf = np.array(r['pos_frames']); fr = np.array(r['frames'])
            print(f'\n[{name}] ori err (deg) over time:', ' '.join(f'{fr[i]}:{of[i]:.0f}' for i in range(0, len(fr), max(1, len(fr) // 12))))
            print(f'[{name}] pos err (cm)  over time:', ' '.join(f'{fr[i]}:{pf[i]*100:.1f}' for i in range(0, len(fr), max(1, len(fr) // 12))))
    json.dump(dict(args=vars(a), anchor=dict(scale=float(s0), resid_mm=float(sc0 * 1000)), body_scale_vs_scan=float(s_b),
                   n_reanchor=n_re, n_fused_corrections=n_corr, step_sources=dict(Counter(src)), results=results), open(out_dir / 'results.json', 'w'), indent=1)
    print(f'done in {time.time() - t_start:.0f}s -> {out_dir}')


if __name__ == '__main__':
    main()
