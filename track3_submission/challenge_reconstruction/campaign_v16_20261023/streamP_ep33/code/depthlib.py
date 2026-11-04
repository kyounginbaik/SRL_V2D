"""Stream P ep33: stereo-depth evidence helpers (eval_stereo_current depth_rgb 1014x760, 65535/(d+1); SAM2 masks of the
source run; final_recon world/camera). All poses in the final_recon WORLD frame unless stated."""
import json, numpy as np, cv2, trimesh
from scipy.spatial import cKDTree
S = '/data1/SRL_V2D/challenge_reconstruction/eval_stereo_current_20261010/episode_000033'
O = '/data1/SRL_V2D/challenge_submissions/track3/eval_hawor_latest_20261008/outputs/episode_000033'
FR = '/data1/SRL_V2D/challenge_reconstruction/final_recon/episode_000033'
EPD = '/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/episodes/episode_000033'
K = json.load(open(O + '/wooden_spoon/anycalib/undistorted_intrinsics.json')); _s = 1014 / K['width']
fx, fy, cx, cy = K['fx'] * _s, K['fy'] * _s, K['cx'] * _s, K['cy'] * _s
SETUP = np.asarray(json.load(open(EPD + '/input_manifest.json'))['setup_from_source'])
MESH = {o: trimesh.load(f'{FR}/assets/{o}_visual.obj', force='mesh') for o in ('blue_cup', 'wooden_spoon')}
SAMP = {o: np.asarray(trimesh.sample.sample_surface_even(m, 20000)[0]) for o, m in MESH.items()}
TREE = {}
def depth(f):
    d = cv2.imread(f'{S}/depth_rgb/{f:06d}.png', -1).astype(np.float64); dep = 65535.0 / d - 1.0; dep[d >= 65535] = np.nan; return dep
def mask(obj, f, track='1'):
    m = cv2.imread(f'{O}/{obj}/masks/{track}/{f:06d}.png', -1); return cv2.resize(m, (1014, 760), interpolation=cv2.INTER_NEAREST) > 0
def backproject(dep, m, wfc):
    m = m & np.isfinite(dep) & (dep > .1) & (dep < 1.5); v, u = np.nonzero(m); Z = dep[v, u]
    P = np.stack([(u - cx) * Z / fx, (v - cy) * Z / fy, Z], 1); return P @ wfc[:3, :3].T + wfc[:3, 3]
def obj_points(obj, f, wfc, erode=2):
    m = mask(obj, f)
    if erode: m = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=erode) > 0
    return backproject(depth(f), m, wfc)
def residual(obj, Pw, T, trim=0.8):
    """data->model distances (mm) of world points Pw to the mesh posed by world_from_object T: median, trimmed mean."""
    if len(Pw) < 50: return np.nan, np.nan
    loc = (Pw - T[:3, 3]) @ T[:3, :3]
    if obj not in TREE: TREE[obj] = cKDTree(SAMP[obj])
    d = TREE[obj].query(loc)[0] * 1000; d.sort(); return float(np.median(d)), float(d[:int(len(d) * trim)].mean())
def silhouette_iou(obj, f, T, wfc):
    """IoU of the projected mesh with the object's SAM2 mask (full res), and the fraction of the mask covered."""
    m = cv2.imread(f'{O}/{obj}/masks/1/{f:06d}.png', -1) > 0; C = np.linalg.inv(wfc) @ T; V = MESH[obj].vertices @ C[:3, :3].T + C[:3, 3]
    Kf = np.array([[K['fx'], 0, K['cx']], [0, K['fy'], K['cy']], [0, 0, 1]]); q = V @ Kf.T; q = q[:, :2] / q[:, 2:]
    r = np.zeros(m.shape, np.uint8); cv2.fillPoly(r, [np.round(q[t]).astype(np.int32) for t in MESH[obj].faces], 1); r = r > 0
    return float((r & m).sum() / max((r | m).sum(), 1)), float((r & m).sum() / max(m.sum(), 1))
def table_plane_z(f, wfc, center_world, radius=0.12):
    """Local table height (setup-frame z) under center_world from stereo points not in any cup/spoon/hand mask within radius
    (setup-frame xy) and |z|<4 cm: robust median z and the RANSAC-free percentile spread."""
    dep = depth(f); excl = np.zeros(dep.shape, bool)
    for o in ('blue_cup', 'wooden_spoon'):
        for tr in ('1', '2', '3'): excl |= cv2.dilate(mask(o, f, tr).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    P = backproject(dep, ~excl, wfc); Ps = P @ SETUP[:3, :3].T + SETUP[:3, 3]; cs = SETUP[:3, :3] @ center_world + SETUP[:3, 3]
    sel = (np.hypot(Ps[:, 0] - cs[0], Ps[:, 1] - cs[1]) < radius) & (np.abs(Ps[:, 2]) < 0.04)
    if sel.sum() < 100: return np.nan, np.nan, int(sel.sum())
    zz = Ps[sel, 2]; return float(np.median(zz)), float(np.percentile(zz, 75) - np.percentile(zz, 25)), int(sel.sum())
