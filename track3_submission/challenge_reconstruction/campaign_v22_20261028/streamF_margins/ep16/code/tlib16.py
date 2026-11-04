"""Stream F ep16: scorer-identical registration + per-frame errors vs target_final_v5 for episode 16 (rack slot 0 = scorer anchor,
wooden_spoon slot 1). Candidate meshes = the shipped component meshes, budget-encoded exactly as make_sub_any.py does."""
import sys, numpy as np, pandas as pd, trimesh
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, '/data1/SRL_V2D/challenge_submissions_SRL/analysis_20261020/scripts')
import public_kit_score as K
M = K.load_metric('MP-SR')
EP = 16
TGT = '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/target_final_v5/episode_000016'
CM = '/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates/N6_ep16_newtrack_z5rack/episode_000016/meshes/episode_000016'
NAMES = ['plastic_dish_rack', 'wooden_spoon']
def H(Rm, t):
    T = np.eye(4); T[:3, :3] = Rm; T[:3, 3] = t; return T
def ref_mesh(n):
    m = trimesh.load(f'{TGT}/{n}_mesh_metric.obj', force='mesh'); m.merge_vertices()
    if len(m.faces) > 40000:
        try: m = m.simplify_quadric_decimation(face_count=40000)
        except Exception: pass
    m.remove_unreferenced_vertices(); return np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64)
_REG = None
def registration():
    """name -> (X, anchor, ca): T_ourbody = T_target @ X; scorer point = our body pos + R ca."""
    global _REG
    if _REG is not None: return _REG
    out = {}
    for slot, n in enumerate(NAMES):
        v, f = K.budget_mesh(f'{CM}/{n}.obj', 4096, 4096)
        cand = M._decode_object_mesh(K.encode(vertices=np.asarray(v, np.float64), faces=np.asarray(f, np.float64)))
        rv, rf = ref_mesh(n); anchor, _ = M._surface_moments(rv, rf)
        ca, rot = M._register_object_body(cand, (rv, rf), anchor, 1_000_003 * EP + 101 * slot)
        out[n] = (H(np.eye(3), anchor) @ np.linalg.inv(H(rot.T, ca)), anchor, ca)
    _REG = out; return out
def target(n): return np.load(f'{TGT}/{n}_trajectory.npz')['T_world_obj']
def target_ourbody(): reg = registration(); return {n: target(n) @ reg[n][0] for n in NAMES}
def mats(df, pre):
    p = df[[f'{pre}_position_{a}' for a in 'xyz']].values; q = df[[f'{pre}_quaternion_{a}' for a in 'xyzw']].values
    T = np.tile(np.eye(4), (len(p), 1, 1)); T[:, :3, :3] = R.from_quat(q).as_matrix(); T[:, :3, 3] = p; return T
def results_tracks(path):
    r = pd.read_parquet(path); r = r[r.environment_id == r.environment_id.min()]
    return {n: mats(g.sort_values('step'), 'achieved') for n, g in r.groupby('object_body_name')}
def errors(Ta):
    """Ta: name -> (n,4,4) our-body poses in any world. Scorer-identical: rotation from the rack frame-0 orientation, offset from the
    rack anchor point. returns name -> (pos cm per frame, deg per frame)."""
    reg = registration(); Tb = target_ourbody(); n = min(len(Ta[NAMES[0]]), len(Tb[NAMES[0]]))
    pt = lambda T, ca: T[:, :3, 3] + np.einsum('nij,j->ni', T[:, :3, :3], ca)
    r0 = NAMES[0]; Rot = Tb[r0][0, :3, :3] @ Ta[r0][0, :3, :3].T
    off = pt(Tb[r0][:1], reg[r0][2])[0] - Rot @ pt(Ta[r0][:1], reg[r0][2])[0]
    out = {}
    for o in NAMES:
        ca = reg[o][2]; pa = pt(Ta[o][:n], ca) @ Rot.T + off; pb = pt(Tb[o][:n], ca)
        Ra = np.einsum('ij,njk->nik', Rot, Ta[o][:n, :3, :3])
        dr = np.degrees(R.from_matrix(np.einsum('nji,njk->nik', Ra, Tb[o][:n, :3, :3])).magnitude())
        out[o] = (np.linalg.norm(pa - pb, axis=1) * 100, dr)
    return out
