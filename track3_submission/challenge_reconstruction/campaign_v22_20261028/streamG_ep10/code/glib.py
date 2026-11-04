"""Stream D (phase 7, v22) helpers, generalised from streamT_C356/code/tlib.py: the scorer's own mesh registration
(MP-SR._register_object_body, same seed/meshes as score_vs_compose.py) to express target_final_v5 poses in OUR mesh body frames."""
import sys, numpy as np, pandas as pd, trimesh
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, '/data1/SRL_V2D/challenge_submissions_SRL/analysis_20261020/scripts')
import public_kit_score as K
M = K.load_metric('MP-SR')
C = '/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023'
CMP = '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/target_final_v5'
V21 = '/data1/SRL_V2D/challenge_submissions/track3_v21_20261024/all_20_episodes_post/track3_submission.parquet'
NAMES = {10: ['blue_cup', 'beige_cup']}   # slot order (object_id) of the v21 package
def H(Rm, t):
    T = np.eye(4); T[:3, :3] = Rm; T[:3, 3] = t; return T
def ref_mesh(ep, n):
    m = trimesh.load(f'{CMP}/episode_{ep:06d}/{n}_mesh_metric.obj', force='mesh'); m.merge_vertices()
    if len(m.faces) > 40000:
        try: m = m.simplify_quadric_decimation(face_count=40000)
        except Exception: pass
    m.remove_unreferenced_vertices(); return np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64)
CA = {}
def registration(ep):
    """name -> 4x4 X with T_ourbody = T_target @ X (X = [I|anchor] @ inv([rot^T|ca])), exactly as the scorer registers."""
    s = pd.read_parquet(V21); s = s[s.row_id.str.contains(f'/e{ep:06d}/')]
    out = {}
    for slot, n in enumerate(NAMES[ep]):
        rows = s[s.row_id.str.endswith(f'/o{slot}')].sort_values('row_id'); enc = rows.object_mesh_b64.fillna('').astype(str)
        cand = M._decode_object_mesh(enc[enc.str.strip() != ''].iloc[0]); v, f = ref_mesh(ep, n)
        anchor, _ = M._surface_moments(v, f)
        ca, rot = M._register_object_body(cand, (v, f), anchor, 1_000_003 * ep + 101 * slot)
        out[n] = H(np.eye(3), anchor) @ np.linalg.inv(H(rot.T, ca)); CA[(ep, n)] = ca
    return out
def target_ourbody(ep):
    X = registration(ep)
    return {n: np.load(f'{CMP}/episode_{ep:06d}/{n}_trajectory.npz')['T_world_obj'] @ X[n] for n in NAMES[ep]}, X
def errs(Ta, Tb, pt=None):
    pt = np.zeros(3) if pt is None else pt[:3]
    pa = np.einsum('nij,j->ni', Ta[:, :3, :3], pt) + Ta[:, :3, 3]; pb = np.einsum('nij,j->ni', Tb[:, :3, :3], pt) + Tb[:, :3, 3]
    dr = np.degrees(np.linalg.norm(R.from_matrix(np.einsum('nji,njk->nik', Ta[:, :3, :3], Tb[:, :3, :3])).as_rotvec(), axis=1))
    return np.linalg.norm(pa - pb, axis=1) * 100, dr
