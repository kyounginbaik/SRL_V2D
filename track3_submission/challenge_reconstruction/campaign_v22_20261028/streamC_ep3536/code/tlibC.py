"""Stream C7 (v22) helpers for eps 35/36 (wooden_spoon): the scorer's own registration of our submitted (budget) mesh onto the
target_final_v5 mesh (seed 1_000_003*ep + 101*slot, as public_kit_score.aligned_errors / score_vs_compose.py), conversion of
target poses into our body frame, scorer-identical per-frame errors (frame-0 rigid alignment).
Adapted (copied) from campaign_v16_20261023/streamT_C356/code/tlib.py and streamW_C36/code/perframe.py."""
import sys, numpy as np, pandas as pd, trimesh
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, '/data1/SRL_V2D/challenge_submissions_SRL/analysis_20261020/scripts')
import public_kit_score as K
M = K.load_metric('MP-SR')
TGT = '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/target_final_v5'
V21 = '/data1/SRL_V2D/challenge_submissions/track3_v21_20261024/all_20_episodes_post/track3_submission.parquet'
C16 = '/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates'
NAME = 'wooden_spoon'
def H(Rm, t):
    T = np.eye(4); T[:3, :3] = Rm; T[:3, 3] = t; return T
def ref_mesh(ep):
    m = trimesh.load(f'{TGT}/episode_{ep:06d}/{NAME}_mesh_metric.obj', force='mesh'); m.merge_vertices()
    if len(m.faces) > 40000:
        try: m = m.simplify_quadric_decimation(face_count=40000)
        except Exception: pass
    m.remove_unreferenced_vertices(); return np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64)
_REG = {}
def registration(ep):
    """returns (X, anchor, ca, rot): T_ourbody = T_target @ X; scorer point = body pos + R ca (ours) / target pos + R anchor."""
    if ep in _REG: return _REG[ep]
    s = pd.read_parquet(V21); s = s[s.row_id.str.contains(f'/e{ep:06d}/')]
    rows = s[s.row_id.str.endswith('/o0')].sort_values('row_id'); enc = rows.object_mesh_b64.fillna('').astype(str)
    cand = M._decode_object_mesh(enc[enc.str.strip() != ''].iloc[0]); v, f = ref_mesh(ep)
    anchor, _ = M._surface_moments(v, f)
    ca, rot = M._register_object_body(cand, (v, f), anchor, 1_000_003 * ep)
    X = H(np.eye(3), anchor) @ np.linalg.inv(H(rot.T, ca)); _REG[ep] = (X, anchor, ca, rot); return _REG[ep]
def target(ep):
    return np.load(f'{TGT}/episode_{ep:06d}/{NAME}_trajectory.npz')['T_world_obj']
def target_ourbody(ep):
    X = registration(ep)[0]; return target(ep) @ X
def mats(df, pre):
    p = df[[f'{pre}_position_{a}' for a in 'xyz']].values; q = df[[f'{pre}_quaternion_{a}' for a in 'xyzw']].values
    T = np.tile(np.eye(4), (len(p), 1, 1)); T[:, :3, :3] = R.from_quat(q).as_matrix(); T[:, :3, 3] = p; return T
def results_track(path):
    r = pd.read_parquet(path); r = r[r.environment_id == r.environment_id.min()]; g = r[r.object_body_name == NAME].sort_values('step')
    return mats(g, 'achieved'), mats(g, 'reference')
def tag_track(tag, ep):
    if tag.startswith('baseline'): return results_track(f'{C16[:-11]}_baseline/episode_{ep:06d}/results/episode_{ep:06d}.parquet')
    return results_track(f'{C16}/{tag}/episode_{ep:06d}/results/episode_{ep:06d}.parquet')
def errors(ep, Ta):
    """scorer-identical per-frame (pos cm, ori deg) of our-body poses Ta (any world) vs target_final_v5 (frame-0 aligned)."""
    X, anchor, ca, rot = registration(ep); Tt = target(ep); n = min(len(Ta), len(Tt))
    Tb = Tt[:n] @ X  # target in our body frame; scorer point = body + R ca for both (identical by construction)
    A = Tb[0] @ np.linalg.inv(Ta[0]); Tal = A @ Ta[:n]
    pa = Tal[:, :3, 3] + np.einsum('nij,j->ni', Tal[:, :3, :3], ca); pb = Tb[:, :3, 3] + np.einsum('nij,j->ni', Tb[:, :3, :3], ca)
    dr = np.degrees(R.from_matrix(np.einsum('nji,njk->nik', Tal[:, :3, :3], Tb[:, :3, :3])).magnitude())
    return np.linalg.norm(pa - pb, axis=1) * 100, dr
def win(x, w=40): return [round(float(x[i:i + w].mean()), 1) for i in range(0, len(x), w)]
