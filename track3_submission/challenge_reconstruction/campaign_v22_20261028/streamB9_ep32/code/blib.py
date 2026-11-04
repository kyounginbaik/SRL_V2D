"""Stream B (v22) helpers: scorer-faithful per-frame errors vs target_final_v5 and target->our-body conversion.
Mirrors public_kit_score.aligned_errors (MP-SR registration + anchor frame-0 alignment; anchor = first object without a symmetry axis
= the spoon) but returns per-frame arrays and accepts arbitrary pose arrays."""
import sys, numpy as np, pandas as pd, trimesh
from pathlib import Path
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, '/data1/SRL_V2D/challenge_submissions_SRL/analysis_20261020/scripts')
import public_kit_score as K
M = K.load_metric('MP-SR')
TGT = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/target_final_v6')
NAMES = ['blue_cup', 'wooden_spoon']
_geo = {}
def geometry(ep):
    if ep in _geo: return _geo[ep]
    g = {}
    for i, n in enumerate(NAMES):
        m = trimesh.load(TGT / f'episode_{ep:06d}/{n}_mesh_metric.obj', force='mesh'); m.merge_vertices()
        if len(m.faces) > 40000:
            try: m = m.simplify_quadric_decimation(face_count=40000)
            except Exception: pass
        m.remove_unreferenced_vertices(); v = np.asarray(m.vertices, np.float64); f = np.asarray(m.faces, np.int64)
        a, _ = M._surface_moments(v, f); ax, o = K.symmetry_axis(n, v)
        g[n] = dict(v=v, f=f, anchor=a, axis=ax, order=int(o))
    _geo[ep] = g; return g
def target(ep):
    return {n: np.load(TGT / f'episode_{ep:06d}/{n}_trajectory.npz')['T_world_obj'].astype(np.float64) for n in NAMES}
def register(ep, mesh_paths):
    """our shipped mesh -> (ca, rot) exactly as the scorer (budgeted 4096 mesh, same seed). Returns dict n->(ca,rot)."""
    g = geometry(ep); out = {}
    for i, n in enumerate(NAMES):
        v, f = K.budget_mesh(str(mesh_paths[n]), 4096, 4096)
        cand = M._decode_object_mesh(K.encode(vertices=np.asarray(v, np.float64), faces=np.asarray(f, np.float64)))
        out[n] = M._register_object_body(cand, (g[n]['v'], g[n]['f']), g[n]['anchor'], 1_000_003 * ep + 101 * i)
    return out
def G_ourbody(ep, reg):
    """4x4 G[n] with T_target_body = T_ourbody @ G  -> T_ourbody = T_target @ inv(G)  (as streamR compose_retarget32_v3)."""
    g = geometry(ep); G = {}
    for n in NAMES:
        ca, rot = reg[n]
        conv = M._convert_object_body(np.array([[[0., 0, 0, 0, 0, 0, 1]]]), ca, rot)[0, 0]; Ra = M._rotation_matrix(conv[3:][None])[0]
        Gi = np.eye(4); Gi[:3, :3] = Ra; Gi[:3, 3] = conv[:3] - Ra @ g[n]['anchor']; G[n] = Gi
    return G
def T2pq(T):
    return np.c_[T[:, :3, 3], R.from_matrix(T[:, :3, :3]).as_quat()]
def per_frame(ep, poses, reg, tgt=None):
    """poses: n -> (N,4,4) in OUR body frames (any world). Returns n -> (pos_cm[N], deg[N]) scorer-aligned vs target."""
    g = geometry(ep); tgt = target(ep) if tgt is None else tgt
    N = min(len(poses[NAMES[0]]), len(tgt[NAMES[0]]))
    ach = np.stack([T2pq(poses[n][:N]) for n in NAMES], 1)[:, None]          # (N,1,S,7)
    ref = np.stack([T2pq(tgt[n][:N]) for n in NAMES], 1)[:, None].copy()
    axes, orders = [], []
    for s, n in enumerate(NAMES):
        ca, rot = reg[n]; ach[:, :, s] = M._convert_object_body(ach[:, :, s], ca, rot)
        ref[:, :, s, :3] += np.einsum('...ij,j->...i', M._rotation_matrix(ref[:, :, s, 3:]), g[n]['anchor'])
        axes.append(g[n]['axis']); orders.append(g[n]['order'])
    ach = M._align_registered_scene(ach, ref, axes)
    pos = np.linalg.norm(ach[..., :3] - ref[..., :3], axis=-1)[:, 0] * 100
    ori = np.degrees(M._symmetry_orientation_errors(ach, ref, axes, orders))[:, 0]
    return {n: (pos[:, s], ori[:, s]) for s, n in enumerate(NAMES)}
def mats(df, pre):
    p = df[[f'{pre}_position_{a}' for a in 'xyz']].values; q = df[[f'{pre}_quaternion_{a}' for a in 'xyzw']].values
    T = np.tile(np.eye(4), (len(p), 1, 1)); T[:, :3, :3] = R.from_quat(q).as_matrix(); T[:, :3, 3] = p; return T
def results_tracks(path):
    r = pd.read_parquet(path); r = r[r.environment_id == r.environment_id.min()]
    a = {n: mats(g.sort_values('step'), 'achieved') for n, g in r.groupby('object_body_name')}
    b = {n: mats(g.sort_values('step'), 'reference') for n, g in r.groupby('object_body_name')}
    return a, b
def comp_meshes(tagdir):
    return {n: next((Path(tagdir) / 'meshes').rglob(f'{n}.obj')) for n in NAMES}
