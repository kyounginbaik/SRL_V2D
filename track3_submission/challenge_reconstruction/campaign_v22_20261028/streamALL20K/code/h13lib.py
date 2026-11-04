"""[E31 copy: scoring target = $E31_TGT (target_E14_HGA / target_E14_HGB)] [H13a copy of F9 mlibF9.py: TGT = target_final_v12b_g15 (G15 retimed b/c clock + camera fix), NAMES + eps 5/35/36/37] [F9 copy of D9 mlib9.py: NAMES + eps 8/20/28/30/38 (scorer slot order)] [D9 copy: target_final_v6, eps 3/6/19/25, optional per-frame MPPE] Stream M8 (v23): scorer-faithful per-frame errors vs target_final_v5 for any episode (generalization of streamB blib.py).
Mirrors public_kit_score.aligned_errors: MP-SR registration of the SHIPPED meshes onto the target meshes (budget 4096, same seed),
anchor frame-0 alignment (_align_registered_scene). Also returns the alignment A so the target can be expressed in the physics world:
T_phys_target_ourbody[k] = inv(A) @ T_target[k] @ Ct @ inv(Cc)  (Ct = H(I, target anchor), Cc = H(rot^T, ca))."""
import sys, numpy as np, trimesh
from pathlib import Path
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, '/data1/SRL_V2D/challenge_submissions_SRL/analysis_20261020/scripts')
import public_kit_score as K
M = K.load_metric('MP-SR')
TGT = Path(__import__('os').environ['E31_TGT'])   # E31: target_E14_HGA / target_E14_HGB
NAMES = {1: ['white_pot', 'white_pot_lid'], 14: ['blue_cup', 'plastic_dish_rack'], 10: ['blue_cup', 'beige_cup'], 8: ['beige_cup', 'white_pot'], 20: ['blue_cup', 'water_pitcher'], 28: ['plastic_dish_rack', 'white_pot'], 30: ['mini_sweeper', 'plastic_dish_rack'], 38: ['mini_dust_pan'], 3: ['white_pot', 'wooden_spoon'], 6: ['white_pot', 'wooden_spoon'], 19: ['mini_sweeper', 'mini_dust_pan'], 25: ['wooden_piece_1', 'wooden_piece_2'], 15: ['plastic_dish_rack', 'wooden_spoon'], 16: ['plastic_dish_rack', 'wooden_spoon'], 32: ['blue_cup', 'wooden_spoon'], 33: ['blue_cup', 'wooden_spoon'], 5: ['white_pot', 'wooden_spoon'], 35: ['wooden_spoon'], 36: ['wooden_spoon'], 37: ['wooden_spoon']}  # ALL20-K: all 20 eps
_geo = {}
def H(Rm, t):
    T = np.eye(4); T[:3, :3] = Rm; T[:3, 3] = t; return T
def geometry(ep):
    if ep in _geo: return _geo[ep]
    g = {}
    for n in NAMES[ep]:
        m = trimesh.load(TGT / f'episode_{ep:06d}/{n}_mesh_metric.obj', force='mesh'); m.merge_vertices()
        if len(m.faces) > 40000:
            try: m = m.simplify_quadric_decimation(face_count=40000)
            except Exception: pass
        m.remove_unreferenced_vertices(); v = np.asarray(m.vertices, np.float64); f = np.asarray(m.faces, np.int64)
        a, _ = M._surface_moments(v, f); ax, o = K.symmetry_axis(n, v)
        g[n] = dict(v=v, f=f, anchor=a, axis=ax, order=int(o))
    _geo[ep] = g; return g
def target(ep):
    return {n: np.load(TGT / f'episode_{ep:06d}/{n}_trajectory.npz')['T_world_obj'].astype(np.float64) for n in NAMES[ep]}
def register(ep, mesh_paths):
    g = geometry(ep); out = {}
    for i, n in enumerate(NAMES[ep]):
        v, f = K.budget_mesh(str(mesh_paths[n]), 4096, 4096)
        cand = M._decode_object_mesh(K.encode(vertices=np.asarray(v, np.float64), faces=np.asarray(f, np.float64)))
        out[n] = M._register_object_body(cand, (g[n]['v'], g[n]['f']), g[n]['anchor'], 1_000_003 * ep + 101 * i)
    return out
def T2pq(T): return np.c_[T[:, :3, 3], R.from_matrix(T[:, :3, :3]).as_quat()]
def pq2T(p):
    T = np.tile(np.eye(4), (len(p), 1, 1)); T[:, :3, :3] = R.from_quat(p[:, 3:]).as_matrix(); T[:, :3, 3] = p[:, :3]; return T
def per_frame(ep, poses, reg, tgt=None, want_A=False, want_mppe=False):
    """poses: n -> (N,4,4) our body frames (any world). Returns n -> (pos_cm[N], deg[N]) (+ A 4x4 physics->target world)."""
    names = NAMES[ep]; g = geometry(ep); tgt = target(ep) if tgt is None else tgt
    N = min(len(poses[names[0]]), len(tgt[names[0]]))
    ach = np.stack([T2pq(poses[n][:N]) for n in names], 1)[:, None]
    ref = np.stack([T2pq(tgt[n][:N]) for n in names], 1)[:, None].copy()
    axes, orders = [], []
    for s, n in enumerate(names):
        ca, rot = reg[n]; ach[:, :, s] = M._convert_object_body(ach[:, :, s], ca, rot)
        ref[:, :, s, :3] += np.einsum('...ij,j->...i', M._rotation_matrix(ref[:, :, s, 3:]), g[n]['anchor'])
        axes.append(g[n]['axis']); orders.append(g[n]['order'])
    al = M._align_registered_scene(ach, ref, axes)
    pos = np.linalg.norm(al[..., :3] - ref[..., :3], axis=-1)[:, 0] * 100
    ori = np.degrees(M._symmetry_orientation_errors(al, ref, axes, orders))[:, 0]
    out = {n: (pos[:, s], ori[:, s]) for s, n in enumerate(names)}
    if want_mppe:   # per-frame MPPE (kit keypoints, symmetry-aware), mean over objects, cm
        pts = M._MPPE_KEYPOINT_DIRECTIONS * M._MPPE_KEYPOINT_RADIUS_M
        out['_mppe'] = np.mean([M._symmetry_point_errors(al[:, :, s], ref[:, :, s], pts, axes[s], orders[s])[:, 0] for s in range(len(names))], 0) * 100
    if want_A:
        A = pq2T(al[:1, 0, 0]) [0] @ np.linalg.inv(pq2T(ach[:1, 0, 0])[0]); return out, A
    return out
def target_phys(ep, reg, A, tgt=None):
    """target poses in OUR body frames in the physics world (n -> (N,4,4))."""
    g = geometry(ep); tgt = target(ep) if tgt is None else tgt; Ai = np.linalg.inv(A); out = {}
    for n in NAMES[ep]:
        ca, rot = reg[n]; Cc = H(rot.T, ca); Ct = H(np.eye(3), g[n]['anchor'])
        out[n] = np.einsum('ij,njk->nik', Ai, tgt[n]) @ Ct @ np.linalg.inv(Cc)
    return out
def comp_meshes(ep, tagdir):
    return {n: next((Path(tagdir) / 'meshes').rglob(f'{n}.obj')) for n in NAMES[ep]}
