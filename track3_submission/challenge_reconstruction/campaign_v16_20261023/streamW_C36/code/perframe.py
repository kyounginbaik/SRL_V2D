"""Per-frame scorer-identical errors (registration + frame-0 alignment) of arbitrary ep36 spoon pose tracks (our mesh body frame,
any world) vs the compose target. Import: perframe.errors(T_list (N,4,4)) -> (pos_cm[N], ori_deg[N])."""
import sys, numpy as np, pandas as pd, trimesh
sys.path.insert(0, '/data1/SRL_V2D/challenge_submissions_SRL/analysis_20261020/scripts')
import public_kit_score as K
M = K.load_metric('MP-SR'); CMP = '/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/compose_target/episode_000036'
TC = np.load(CMP + '/wooden_spoon_trajectory.npz')['T_world_obj']
OURMESH = '/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep36/source_corrected/episode_000036/assets/wooden_spoon_visual.obj'
def _mesh(p):
    m = trimesh.load(p, force='mesh'); m.merge_vertices()
    if len(m.faces) > 40000:
        try: m = m.simplify_quadric_decimation(face_count=40000)
        except Exception: pass
    m.remove_unreferenced_vertices(); return np.asarray(m.vertices, float), np.asarray(m.faces, np.int64)
rv, rf = _mesh(CMP + '/wooden_spoon_mesh_metric.obj'); anchor, _ = M._surface_moments(rv, rf)
cm = trimesh.load(OURMESH, force='mesh'); cv, cf = np.asarray(cm.vertices, float), np.asarray(cm.faces, np.int64)
ca, brot = M._register_object_body((cv, cf), (rv, rf), anchor, 1_000_003 * 36 + 0)
def tq(T):
    from scipy.spatial.transform import Rotation as R
    return np.concatenate([T[:, :3, 3], R.from_matrix(T[:, :3, :3]).as_quat()], 1)
def errors(T):
    n = len(T); a = tq(T)[:, None, None, :].copy(); r = tq(TC[:n])[:, None, None, :].copy()
    a[:, :, 0] = M._convert_object_body(a[:, :, 0], ca, brot)
    r[:, :, 0, :3] += np.einsum('...ij,j->...i', M._rotation_matrix(r[:, :, 0, 3:]), anchor)
    a = M._align_registered_scene(a, r, [np.zeros(3)])
    pos = np.linalg.norm(a[..., :3] - r[..., :3], axis=-1)[:, 0, 0] * 100
    ori = np.degrees(M._symmetry_orientation_errors(a, r, [np.zeros(3)], [1])[:, 0, 0]); return pos, ori
def compose_in_our_body():
    """compose spoon poses expressed for OUR mesh body frame in the compose world: T_c @ G, G = our body -> their body."""
    from scipy.spatial.transform import Rotation as R
    G = np.eye(4); G[:3, :3] = brot.T; G[:3, 3] = anchor - brot.T @ (np.zeros(3)) if False else 0
    return None
def Gc():
    """T_our_body = T_compose @ Gc (scorer registration: their anchor frame == our body pose @ [brot^T, ca])."""
    H = np.eye(4); H[:3, :3] = brot.T; H[:3, 3] = ca
    Ta = np.eye(4); Ta[:3, 3] = anchor; return Ta @ np.linalg.inv(H)
