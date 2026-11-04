"""Stream T helpers: the scorer's own mesh registration (MP-SR._register_object_body, same seed/meshes as score_vs_compose.py) to
express the compose_target poses in OUR mesh body frames, and per-frame comparison tools."""
import sys, numpy as np, pandas as pd, trimesh
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, '/data1/SRL_V2D/challenge_submissions_SRL/analysis_20261020/scripts')
import public_kit_score as K
M = K.load_metric('MP-SR')
C = '/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023'
CMP = C + '/compose_target'
V21 = '/data1/SRL_V2D/challenge_submissions/track3_v21_20261024/all_20_episodes_post/track3_submission.parquet'
NAMES = ['white_pot', 'wooden_spoon']
def H(Rm, t):
    T = np.eye(4); T[:3, :3] = Rm; T[:3, 3] = t; return T
def ref_mesh(ep, n):
    m = trimesh.load(f'{CMP}/episode_{ep:06d}/{n}_mesh_metric.obj', force='mesh'); m.merge_vertices()
    if len(m.faces) > 40000:
        try: m = m.simplify_quadric_decimation(face_count=40000)
        except Exception: pass
    m.remove_unreferenced_vertices(); return np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64)
def registration(ep):
    """per object: (anchor, ca, rot) exactly as aligned_errors()/the scorer compute them; returns dict name -> 4x4 X with
    T_ourbody = T_compose @ X  (X = [I|anchor] @ inv([rot^T|ca]))."""
    s = pd.read_parquet(V21); s = s[s.row_id.str.contains(f'/e{ep:06d}/')]
    out = {}
    for slot, n in enumerate(NAMES):
        rows = s[s.row_id.str.endswith(f'/o{slot}')].sort_values('row_id'); enc = rows.object_mesh_b64.fillna('').astype(str)
        cand = M._decode_object_mesh(enc[enc.str.strip() != ''].iloc[0]); v, f = ref_mesh(ep, n)
        anchor, _ = M._surface_moments(v, f)
        ca, rot = M._register_object_body(cand, (v, f), anchor, 1_000_003 * ep + 101 * slot)
        out[n] = H(np.eye(3), anchor) @ np.linalg.inv(H(rot.T, ca)); CA[(ep, n)] = ca
    return out
CA = {}
def compose_ourbody(ep):
    X = registration(ep)
    return {n: np.load(f'{CMP}/episode_{ep:06d}/{n}_trajectory.npz')['T_world_obj'] @ X[n] for n in NAMES}, X
def mats(df, pre):
    p = df[[f'{pre}_position_{a}' for a in 'xyz']].values; q = df[[f'{pre}_quaternion_{a}' for a in 'xyzw']].values
    T = np.tile(np.eye(4), (len(p), 1, 1)); T[:, :3, :3] = R.from_quat(q).as_matrix(); T[:, :3, 3] = p; return T
def results_tracks(path):
    r = pd.read_parquet(path); r = r[r.environment_id == r.environment_id.min()]
    a = {n: mats(g.sort_values('step'), 'achieved') for n, g in r.groupby('object_body_name')}
    b = {n: mats(g.sort_values('step'), 'reference') for n, g in r.groupby('object_body_name')}
    return a, b
def anchor_points(ep):
    """the scorer's position = body position + R @ anchor-in-our-body; return anchor in our body frame per object."""
    if (ep, NAMES[0]) not in CA: registration(ep)
    return {n: CA[(ep, n)] for n in NAMES}
def errs(Ta, Tb, pt=None):
    """per-frame position (cm, of body point pt or origin) and rotation (deg) differences"""
    pt = np.zeros(3) if pt is None else pt[:3]
    pa = np.einsum('nij,j->ni', Ta[:, :3, :3], pt) + Ta[:, :3, 3]; pb = np.einsum('nij,j->ni', Tb[:, :3, :3], pt) + Tb[:, :3, 3]
    dr = np.degrees(np.linalg.norm(R.from_matrix(np.einsum('nji,njk->nik', Ta[:, :3, :3], Tb[:, :3, :3])).as_rotvec(), axis=1))
    return np.linalg.norm(pa - pb, axis=1) * 100, dr
