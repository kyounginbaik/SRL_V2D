"""Stream F (v22) ep19 helpers, copy of streamV_C19/code/common19.py with CMP = target_final_v5 (scorer registration of our packed meshes, scorer-mirroring errors)."""
# poses expressed in our mesh body frames, and a scorer-mirroring error (anchor = mini_sweeper frame-0 full SE3 alignment).
import sys, json, hashlib, numpy as np, pandas as pd, trimesh
from pathlib import Path
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, '/data1/SRL_V2D/challenge_submissions_SRL/analysis_20261020/scripts')
import public_kit_score as K
M = K.load_metric('MP-SR')
C16 = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023'); V = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamF_margins/ep19')
CMP = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/target_final_v5/episode_000019'); EP = 19; NAMES = ['mini_sweeper', 'mini_dust_pan']
V21PRE = Path('/data1/SRL_V2D/challenge_submissions/track3_v21_20261024/all_20_episodes/track3_submission.parquet')
SRC = Path('/data1/SRL_V2D/challenge_reconstruction/final_recon/episode_000019')
EPDIR = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/expanded_batch/repaired_export_inputs/episode_000019')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref_mesh(n):
    m = trimesh.load(CMP / f'{n}_mesh_metric.obj', force='mesh'); m.merge_vertices()
    if len(m.faces) > 40000:
        try: m = m.simplify_quadric_decimation(face_count=40000)
        except Exception: pass
    m.remove_unreferenced_vertices(); return np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64)
def packed_meshes(parquet=V21PRE):
    s = pd.read_parquet(parquet, columns=['row_id', 'object_mesh_b64']); s = s[s.row_id.str.contains('/e000019/')]
    out = {}
    for slot in range(2):
        enc = s[s.row_id.str.endswith(f'/o{slot}')].object_mesh_b64.fillna('').astype(str); out[NAMES[slot]] = M._decode_object_mesh(enc[enc.str.strip() != ''].iloc[0])
    return out
def registration(packed):
    """Mreg[n]: our body pose = compose pose @ Mreg (exact inverse of the scorer's _convert_object_body with its seed)."""
    out = {}
    for slot, n in enumerate(NAMES):
        v, f = ref_mesh(n); anchor, _ = M._surface_moments(v, f)
        ca, rot = M._register_object_body(packed[n], (v, f), anchor, 1_000_003 * EP + 101 * slot)
        T = np.eye(4); T[:3, :3] = rot; T[:3, 3] = anchor - rot @ ca
        cv = np.asarray(packed[n][0]); P = cv @ T[:3, :3].T + T[:3, 3]   # our mesh in the compose body frame
        from scipy.spatial import cKDTree
        rs = trimesh.sample.sample_surface(trimesh.Trimesh(v, f, process=False), 20000, seed=0)[0]
        cs = trimesh.sample.sample_surface(trimesh.Trimesh(P, np.asarray(packed[n][1]), process=False), 20000, seed=0)[0]
        ch = 0.5 * (cKDTree(rs).query(cs)[0].mean() + cKDTree(cs).query(rs)[0].mean())
        out[n] = dict(M=T, candidate_anchor=ca, rot=rot, anchor=anchor, chamfer_mm=float(ch * 1000))
    return out
def compose_tracks():
    return {n: np.load(CMP / f'{n}_trajectory.npz')['T_world_obj'].astype(float) for n in NAMES}
def to_T(p, q_wxyz):
    T = np.tile(np.eye(4), (len(p), 1, 1)); T[:, :3, :3] = R.from_quat(np.asarray(q_wxyz)[:, [1, 2, 3, 0]]).as_matrix(); T[:, :3, 3] = p; return T
def scorer_errors(ach, ref, reg):
    """ach, ref: dict name -> (T,4,4) poses of OUR mesh body (ach) and the compose body (ref). Mirrors aligned_errors."""
    A, Rf = {}, {}
    for n in NAMES:
        r = reg[n]; a = ach[n].copy(); a[:, :3, 3] += np.einsum('tij,j->ti', a[:, :3, :3], r['candidate_anchor']); a[:, :3, :3] = a[:, :3, :3] @ r['rot'].T
        b = ref[n].copy(); b[:, :3, 3] += np.einsum('tij,j->ti', b[:, :3, :3], r['anchor']); A[n], Rf[n] = a, b
    Ra = Rf[NAMES[0]][0, :3, :3] @ A[NAMES[0]][0, :3, :3].T; off = Rf[NAMES[0]][0, :3, 3] - Ra @ A[NAMES[0]][0, :3, 3]
    out = {}
    n_ = min(len(A[NAMES[0]]), len(Rf[NAMES[0]]))
    for n in NAMES:
        p = A[n][:n_, :3, 3] @ Ra.T + off; Rr = np.einsum('ij,tjk->tik', Ra, A[n][:n_, :3, :3])
        e = np.linalg.norm(p - Rf[n][:n_, :3, 3], axis=1) * 100; d = np.degrees(R.from_matrix(np.einsum('tji,tjk->tik', Rr, Rf[n][:n_, :3, :3])).magnitude())
        out[n] = dict(cm=float(e.mean()), deg=float(d.mean()), cm_max=float(e.max()), deg_max=float(d.max()), cm_t=e, deg_t=d)
    return out
_REG = None
def compose_score_traj(traj):
    """Scorer-mirroring errors of a rollout trajectory.npz (setup world) vs the compose target (no relabel)."""
    global _REG
    if _REG is None: _REG = registration(packed_meshes())
    t = np.load(traj); on = list(t['object_names']); comp = compose_tracks()
    ach = {n: to_T(t['object_positions'][:, on.index(n)], t['object_quaternions_wxyz'][:, on.index(n)]) for n in NAMES}
    e = scorer_errors(ach, comp, _REG); return {n: (round(e[n]['cm'], 3), round(e[n]['deg'], 2)) for n in NAMES}, e
