"""H2H other (phase 5): shared loaders for OUR reference tracks and the coworker's NEW compose_eval tracks, plus the
independent evidence (FoundationStereo depth in the rectified ego_cam_c, SAM2 object/hand masks in the AnyCalib-undistorted
ego_cam_a image). All comparisons are done in the PHYSICAL ego_cam_a frame ("A") per frame:
  ours : A_T_obj = inv(world_from_rgb[t]) @ world_from_object[t]        (our 'rgb' camera = undistorted ego_cam_a)
  new  : A_T_obj = A_T_rect @ inv(cam0_T_cam[t]) @ T_world_obj[t]      (their camera = rectified stereo left)
  A_T_rect = inv(a_to_left) @ c_to_left @ R1^T (public calibration + the episode's rectification.npz) -- verified by hand-eye
  (camcheck.py)."""
import json, numpy as np, cv2, trimesh
from pathlib import Path
from scipy.spatial import cKDTree
CR = Path('/data1/SRL_V2D/challenge_reconstruction'); C16 = CR / 'campaign_v16_20261023'
NEWROOT = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/h2h_other/new_tracks_copy')   # verbatim copy of the compose_eval_20261021 extract (scratchpad /tmp is node-local)
HAW = Path('/data1/SRL_V2D/challenge_submissions/track3/eval_hawor_latest_20261008/outputs')
SCANS = Path('/data1/SRL_V2D/gt_best_per_object')
CAL = json.load(open('/data1/video_to_data_challenge/track_3/public/meta/camera_calibration.json'))['cameras']
KA = np.array(CAL['ego_cam_a']['intrinsic_matrix'], float); DA = np.array(CAL['ego_cam_a']['distortion_coefficients'], float)
T_AC = np.linalg.inv(np.array(CAL['ego_cam_a']['extrinsics_to_stereo_left'])) @ np.array(CAL['ego_cam_c']['extrinsics_to_stereo_left'])
OURS_COMPONENT = {
    1: C16 / 'streamH_ep1/staging/stage_prev_s128_fg0.003_c512/components/episode_000001',
    8: CR / 'dexagent_final_recon_20261017/refinement_20261020_pot/staging_v2/components/episode_000008',
    10: C16 / 'streamL_ep10/staging_d/components/episode_000010',
    14: C16 / 'streamM_ep14/staging_6prong/components/episode_000014',
    19: C16 / 'streamC_rigid_ilc/ep19/staging/components/episode_000019',
    20: CR / 'dexagent_remaining7_20261019/components/episode_000020',
    25: C16 / 'streamQ_ep25/staging_w3_L47/components/episode_000025',
    28: C16 / 'streamO_ep28/staging_6prong_C710/components/episode_000028',
    30: C16 / 'streamI_ep30/staging_6prong/components/episode_000030',
    38: CR / 'dexagent_final_recon_20261017/refinement_20261021_ep38/staging/components/episode_000038'}
EP19_RELABEL = CR / 'recon_v15_20261022/episode_000019/change.json'   # applied in the v20 post-process to the sweeper poses
def stereo_dir(ep):
    return CR / ('pose_ablation/episode01_stereo_20261010' if ep == 1 else f'eval_stereo_current_20261010/episode_{ep:06d}')
def load_ours(ep):
    src = next((OURS_COMPONENT[ep] / 'portable/source').iterdir()); d = np.load(src / 'object_poses.npz', allow_pickle=True)
    names = [str(n) for n in d['names']]; W = d['world_from_object'].astype(float); Wc = d['world_from_rgb'].astype(float)
    if ep == 19:
        ch = json.load(open(EP19_RELABEL)); G = np.array(ch['G']); W = W.copy(); W[:, ch['slot']] = W[:, ch['slot']] @ G
    cam = np.einsum('tij,tojk->toik', np.linalg.inv(Wc), W)
    meshes = {n: trimesh.load(src / f'assets/{n}.obj', force='mesh', process=False) for n in names}
    return dict(names=names, A_T_obj=cam, world=W, world_from_cam=Wc, meshes=meshes, src=str(src))
def A_T_rect(ep):
    r = np.load(stereo_dir(ep) / 'rectification.npz'); M = np.eye(4); M[:3, :3] = r['R1'].T; return T_AC @ M
def load_new(ep, names):
    e = NEWROOT / f'episode_{ep:06d}'; C = np.load(e / 'camera_motion.npz')['cam0_T_cam'].astype(float); X = A_T_rect(ep)
    out = {}; meshes = {}
    for n in names:
        p = e / f'{n}_trajectory.npz'
        if not p.exists(): continue
        T = np.load(p)['T_world_obj'].astype(float); out[n] = np.einsum('ij,tjk->tik', X, np.einsum('tij,tjk->tik', np.linalg.inv(C), T))
        meshes[n] = trimesh.load(e / f'{n}_mesh_metric.obj', force='mesh', process=False)
    return dict(names=[n for n in names if n in out], A_T_obj=out, cam0_T_cam=C, meshes=meshes)
# ---------------- evidence ----------------
class Evidence:
    def __init__(self, ep, names, mask_override=None):
        self.ep = ep; self.sd = stereo_dir(ep); r = np.load(self.sd / 'rectification.npz'); self.R1 = r['R1']; P = r['P1']
        y, x = np.indices((800, 1280)); self.rays = np.stack([(x - P[0, 2]) / P[0, 0], (y - P[1, 2]) / P[1, 1], np.ones_like(x, float)], -1).reshape(-1, 3)
        any_ = HAW / f'episode_{ep:06d}' / names[0] / 'anycalib'
        def mat(p): a = json.load(open(p)); return np.array([[a['fx'], 0, a['cx']], [0, a['fy'], a['cy']], [0, 0, 1.]])
        self.Kany = mat(any_ / 'intrinsics.json'); self.Dany = np.array(json.load(open(any_ / 'distortion.json'))['params'])
        self.Kout = mat(any_ / 'undistorted_intrinsics.json'); u = json.load(open(any_ / 'undistorted_intrinsics.json')); self.size = (u['width'], u['height'])
        self.names = names; self.mask_dirs = {n: HAW / f'episode_{ep:06d}' / n / 'masks' for n in names}
        if mask_override: self.mask_dirs.update(mask_override)
    def project(self, P):
        """A-frame points -> pixel in the AnyCalib-undistorted full-res image (exactly as project_depth.py)."""
        P = np.asarray(P, float); uv = np.full((len(P), 2), np.nan); ok = P[:, 2] > 1e-3
        if ok.any():
            q = cv2.projectPoints(P[ok], np.zeros(3), np.zeros(3), KA, DA)[0]; q = cv2.fisheye.undistortPoints(q, self.Kany, self.Dany, P=self.Kout)[:, 0]; uv[ok] = q
        return uv
    def stereo_points(self, t, zmin=.12, zmax=1.6):
        f = self.sd / f'stereo/{t:06d}.npz'
        if not f.exists(): return None, None
        z = np.load(f)['depth_m'].ravel(); v = (z > zmin) & (z < zmax); pts = self.rays[v] * z[v, None]; pts = pts @ self.R1
        pts = pts @ T_AC[:3, :3].T + T_AC[:3, 3]; uv = self.project(pts); ok = np.isfinite(uv).all(1); return pts[ok], uv[ok]
    def mask(self, n, t, track='1'):
        d = self.mask_dirs[n]; p = (d / track / f'{t:06d}.png') if (d / track).exists() else (d / f'{t:06d}.png')
        m = cv2.imread(str(p), 0)
        if m is None: return None
        if m.shape[::-1] != self.size: m = cv2.resize(m, self.size, interpolation=cv2.INTER_NEAREST)
        return m > 0
    def hands(self, t):
        h = np.zeros(self.size[::-1], bool)
        d = HAW / f'episode_{self.ep:06d}' / self.names[0] / 'masks'
        for tr in ('2', '3', '4'):
            p = d / tr / f'{t:06d}.png'
            if p.exists():
                m = cv2.imread(str(p), 0)
                if m is not None:
                    if m.shape[::-1] != self.size: m = cv2.resize(m, self.size, interpolation=cv2.INTER_NEAREST)
                    h |= m > 0
        return h
def render_sil(ev, mesh, A_T, scale=0.5):
    V = np.asarray(mesh.vertices) @ A_T[:3, :3].T + A_T[:3, 3]
    if (V[:, 2] < 0.05).mean() > 0.01: return None
    uv = ev.project(V) * scale; W, H = int(ev.size[0] * scale), int(ev.size[1] * scale); s = np.zeros((H, W), np.uint8)
    F = np.asarray(mesh.faces); tri = uv[F]; ok = np.isfinite(tri).all((1, 2)); tri = np.round(tri[ok]).astype(np.int32)
    # per-triangle convex fill (cv2.fillPoly with many contours applies even-odd filling and punches holes)
    area = np.abs((tri[:, 1, 0] - tri[:, 0, 0]) * (tri[:, 2, 1] - tri[:, 0, 1]) - (tri[:, 2, 0] - tri[:, 0, 0]) * (tri[:, 1, 1] - tri[:, 0, 1]))
    for tr in tri[area > 0]: cv2.fillConvexPoly(s, tr, 1)
    for p in tri[area == 0][:, 0]:
        if 0 <= p[0] < W and 0 <= p[1] < H: s[p[1], p[0]] = 1
    return s > 0
class MeshDist:
    def __init__(self, mesh, n=150000):
        self.s = trimesh.sample.sample_surface(mesh, n, seed=0)[0]; self.tree = cKDTree(self.s)
    def d(self, P_A, A_T):
        loc = (P_A - A_T[:3, 3]) @ A_T[:3, :3]; return self.tree.query(loc)[0]
def visible_model_points(md, A_T, ev, mask_full):
    """model->data: mesh sample points facing the camera are not tested (no normals); instead return the posed surface samples
    that project inside the object mask (used for coverage)."""
    pass
def sil_mesh(mesh, max_faces=20000):
    if len(mesh.faces) <= max_faces: return mesh
    try: return mesh.simplify_quadric_decimation(face_count=max_faces)
    except Exception:
        return trimesh.Trimesh(mesh.vertices, np.asarray(mesh.faces)[::int(np.ceil(len(mesh.faces) / max_faces))], process=False)
