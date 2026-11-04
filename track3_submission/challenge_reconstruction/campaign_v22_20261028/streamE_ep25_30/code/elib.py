"""Stream E helpers (copied/adapted from campaign_v16 streamT_C356/code/tlib.py): the scorer's own mesh registration
(MP-SR._register_object_body, same seed as score_vs_compose.py) of a component's OWN mesh onto the target_final_v5 mesh, to
express target poses in our mesh body frames; plus pose-diff tools."""
import sys, numpy as np, pandas as pd, trimesh
from pathlib import Path
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, '/data1/SRL_V2D/challenge_submissions_SRL/analysis_20261020/scripts')
import public_kit_score as K
M = K.load_metric('MP-SR')
TGT = '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/target_final_v5'
def H(Rm, t):
    T = np.eye(4); T[:3, :3] = Rm; T[:3, 3] = t; return T
def ref_mesh(ep, n, tgt=TGT):
    m = trimesh.load(f'{tgt}/episode_{ep:06d}/{n}_mesh_metric.obj', force='mesh'); m.merge_vertices()
    if len(m.faces) > 40000:
        try: m = m.simplify_quadric_decimation(face_count=40000)
        except Exception: pass
    m.remove_unreferenced_vertices(); return np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64)
def cand_mesh(path):
    v, f = K.budget_mesh(str(path), 4096, 4096)
    return M._decode_object_mesh(K.encode(vertices=np.asarray(v, np.float64), faces=np.asarray(f, np.float64)))
def registration(ep, slot, n, our_mesh_path, seed=None, tgt=TGT):
    """X with T_ourbody = T_target @ X, exactly as aligned_errors()/the scorer register (seed 1_000_003*ep+101*slot)."""
    cand = cand_mesh(our_mesh_path); v, f = ref_mesh(ep, n, tgt)
    anchor, _ = M._surface_moments(v, f)
    ca, rot = M._register_object_body(cand, (v, f), anchor, 1_000_003 * ep + 101 * slot if seed is None else seed)
    return H(np.eye(3), anchor) @ np.linalg.inv(H(rot.T, ca)), ca, rot, anchor
def target(ep, n, tgt=TGT):
    return np.load(f'{tgt}/episode_{ep:06d}/{n}_trajectory.npz')['T_world_obj']
def mats(df, pre):
    p = df[[f'{pre}_position_{a}' for a in 'xyz']].values; q = df[[f'{pre}_quaternion_{a}' for a in 'xyzw']].values
    T = np.tile(np.eye(4), (len(p), 1, 1)); T[:, :3, :3] = R.from_quat(q).as_matrix(); T[:, :3, 3] = p; return T
def results_tracks(path):
    r = pd.read_parquet(path); r = r[r.environment_id == r.environment_id.min()]
    a = {n: mats(g.sort_values('step'), 'achieved') for n, g in r.groupby('object_body_name')}
    b = {n: mats(g.sort_values('step'), 'reference') for n, g in r.groupby('object_body_name')}
    return a, b
def errs(Ta, Tb, pt=None):
    pt = np.zeros(3) if pt is None else pt[:3]
    pa = np.einsum('nij,j->ni', Ta[:, :3, :3], pt) + Ta[:, :3, 3]; pb = np.einsum('nij,j->ni', Tb[:, :3, :3], pt) + Tb[:, :3, 3]
    dr = np.degrees(np.linalg.norm(R.from_matrix(np.einsum('nji,njk->nik', Ta[:, :3, :3], Tb[:, :3, :3])).as_rotvec(), axis=1))
    return np.linalg.norm(pa - pb, axis=1) * 100, dr
