"""Kit-faithful track-vs-track errors (registration of achieved mesh onto reference mesh, anchor frame-0 scene alignment, symmetry-aware
orientation), using the unchanged kit functions. Per-frame errors are returned so time windows can be analysed."""
import sys, numpy as np, trimesh
sys.path.insert(0, '/data1/SRL_V2D/challenge_submissions_SRL/analysis_20261020/scripts')
import public_kit_score as K
from v2dlb.mesh_budget import budget_mesh
from scipy.spatial.transform import Rotation as Rot
M = K.load_metric('MP-SR')
def T2p(T):
    q = Rot.from_matrix(T[..., :3, :3].reshape(-1, 3, 3)).as_quat().reshape(T.shape[:-2] + (4,))
    return np.concatenate([T[..., :3, 3], q], -1)
_cache = {}
def ref_mesh(path, name):
    key = ('r', path)
    if key not in _cache:
        m = trimesh.load(path, force='mesh'); m.merge_vertices()
        if len(m.faces) > 40000:
            try: m = m.simplify_quadric_decimation(face_count=40000)
            except Exception: pass
        m.remove_unreferenced_vertices(); v = np.asarray(m.vertices, np.float64); f = np.asarray(m.faces, np.int64)
        a, _ = M._surface_moments(v, f); ax, o = K.symmetry_axis(name, v)
        _cache[key] = (v, f, a, ax, o)
    return _cache[key]
def sub_mesh(path):
    key = ('s', path)
    if key not in _cache:
        v, f = budget_mesh(path, 4096, 4096); _cache[key] = (np.asarray(v, np.float64), np.asarray(f, np.int64))
    return _cache[key]
def errors(ach, ref, ep):
    """ach/ref: list of (name, T[F,4,4], mesh_path) in slot order. Returns pos_cm[F,n], ori_deg[F,n] after kit alignment."""
    F = min(len(a[1]) for a in ach + ref)
    A = np.stack([T2p(a[1][:F]) for a in ach], 1)[:, None]; R = np.stack([T2p(r[1][:F]) for r in ref], 1)[:, None]
    axes, orders = [], []
    for s, ((n, _, pa), (_, _, pr)) in enumerate(zip(ach, ref)):
        v, f, anc, ax, o = ref_mesh(pr, n)
        ca, rot = M._register_object_body(sub_mesh(pa), (v, f), anc, 1_000_003 * ep + 101 * s)
        A[:, :, s] = M._convert_object_body(A[:, :, s], ca, rot)
        R[:, :, s, :3] += np.einsum('...ij,j->...i', M._rotation_matrix(R[:, :, s, 3:]), anc)
        axes.append(ax); orders.append(o)
    A = M._align_registered_scene(A, R, axes)
    pos = np.linalg.norm(A[..., :3] - R[..., :3], axis=-1)[:, 0] * 100
    ori = np.degrees(M._symmetry_orientation_errors(A, R, axes, orders))[:, 0]
    return pos, ori
