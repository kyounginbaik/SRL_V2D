"""Fast offline mirror of score_vs_compose.py for ep8 (kit functions: registration, body conversion, frame-0 scene alignment,
symmetry-aware orientation). score(P) with P = (T, 2, 4, 4) object poses (beige_cup, white_pot) in ANY world, our body frames.
CLI: uscore.py RESULTS_PARQUET  (achieved poses of a rollout)  -> per-object mean cm / deg (+ per-second bins)."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, trimesh
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, '/data1/SRL_V2D/challenge_submissions_SRL/analysis_20261020/scripts')
import public_kit_score as K
C16 = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023'); CMP = C16 / 'compose_target/episode_000008'; EP = 8
SUB = C16 / 'streamU_C8/scores/v21_baseline/sub.parquet'
NAMES = ['beige_cup', 'white_pot']
M = K.load_metric('MP-SR')
_sub = pd.read_parquet(SUB); _sub = _sub[_sub.row_id.str.contains('/e000008/')]
GEO = []
for slot, n in enumerate(NAMES):
    enc = _sub[_sub.row_id.str.endswith(f'/o{slot}')].object_mesh_b64.fillna('').astype(str); cand = M._decode_object_mesh(enc[enc.str.strip() != ''].iloc[0])
    m = trimesh.load(CMP / f'{n}_mesh_metric.obj', force='mesh'); m.merge_vertices()
    if len(m.faces) > 40000:
        try: m = m.simplify_quadric_decimation(face_count=40000)
        except Exception: pass
    m.remove_unreferenced_vertices(); v = np.asarray(m.vertices, np.float64); f = np.asarray(m.faces, np.int64)
    anchor, _ = M._surface_moments(v, f); ca, rot = M._register_object_body(cand, (v, f), anchor, 1_000_003 * EP + 101 * slot)
    ax, order = K.symmetry_axis(n, v); GEO.append(dict(ca=ca, rot=rot, anchor=anchor, axis=ax, order=order))
REF = np.stack([np.load(CMP / f'{n}_trajectory.npz')['T_world_obj'].astype(float) for n in NAMES], 1)
def to7(T):
    q = R.from_matrix(T[..., :3, :3].reshape(-1, 3, 3)).as_quat().reshape(T.shape[:-2] + (4,)); return np.concatenate([T[..., :3, 3], q], -1)
def score(P, per_frame=False):
    n = min(len(P), len(REF)); a = to7(P[:n])[:, None].copy(); r = to7(REF[:n])[:, None].copy()
    for s, g in enumerate(GEO):
        a[:, :, s] = M._convert_object_body(a[:, :, s], g['ca'], g['rot'])
        r[:, :, s, :3] += np.einsum('...ij,j->...i', M._rotation_matrix(r[:, :, s, 3:]), g['anchor'])
    axes = [g['axis'] for g in GEO]; orders = [g['order'] for g in GEO]
    a = M._align_registered_scene(a, r, axes)
    pos = np.linalg.norm(a[..., :3] - r[..., :3], axis=-1)[:, 0] * 100; ori = np.degrees(M._symmetry_orientation_errors(a, r, axes, orders))[:, 0]
    out = {nm: (round(float(pos[:, i].mean()), 3), round(float(ori[:, i].mean()), 2)) for i, nm in enumerate(NAMES)}
    return (out, pos, ori) if per_frame else out
def load_results(p):
    d = pd.read_parquet(p); T = []
    for n in NAMES:
        x = d[d.object_body_name == n].sort_values('step'); q = x[['achieved_quaternion_x', 'achieved_quaternion_y', 'achieved_quaternion_z', 'achieved_quaternion_w']].to_numpy()
        M4 = np.tile(np.eye(4), (len(x), 1, 1)); M4[:, :3, :3] = R.from_quat(q).as_matrix(); M4[:, :3, 3] = x[['achieved_position_x', 'achieved_position_y', 'achieved_position_z']].to_numpy(); T.append(M4)
    return np.stack(T, 1)
def score_own(P, REFP, per_frame=False):
    """Same alignment/metric but against a reference given in OUR body frames (no registration); cup axis from our mesh."""
    n = min(len(P), len(REFP)); a = to7(P[:n])[:, None].copy(); r = to7(REFP[:n])[:, None].copy()
    axes = [OWN_AXIS, np.zeros(3)]; orders = [360, 1]
    for s in range(2):
        for x in (a, r): x[:, :, s, :3] += np.einsum('...ij,j->...i', M._rotation_matrix(x[:, :, s, 3:]), OWN_ANCHOR[s])
    a = M._align_registered_scene(a, r, axes)
    pos = np.linalg.norm(a[..., :3] - r[..., :3], axis=-1)[:, 0] * 100; ori = np.degrees(M._symmetry_orientation_errors(a, r, axes, orders))[:, 0]
    out = {nm: (round(float(pos[:, i].mean()), 3), round(float(ori[:, i].mean()), 2)) for i, nm in enumerate(NAMES)}
    return (out, pos, ori) if per_frame else out
_om = [trimesh.load(f'/data1/SRL_V2D/challenge_reconstruction/final_recon/episode_000008/assets/{n}.obj', force='mesh') for n in NAMES]
OWN_ANCHOR = [M._surface_moments(np.asarray(m.vertices, float), np.asarray(m.faces, np.int64))[0] for m in _om]
OWN_AXIS = K.symmetry_axis('beige_cup', np.asarray(_om[0].vertices))[0]
def bins(pos, ori, step=20):
    return ['t=%4.1fs cup %.2f cm %.1f deg | pot %.2f cm %.1f deg' % (s / 20, pos[s:s+step, 0].mean(), ori[s:s+step, 0].mean(), pos[s:s+step, 1].mean(), ori[s:s+step, 1].mean()) for s in range(0, len(pos), step)]
def load_traj(p, fps=20.):
    """execution/trajectory.npz (setup frame; any rigid frame is fine for this metric) resampled to the 20 fps reference frames."""
    d = np.load(p); names = [str(x) for x in d['object_names']]; ts = d['timestamps_seconds']; P = d['object_positions']; Q = d['object_quaternions_wxyz']
    if P.ndim == 4: P, Q = P[:, 0], Q[:, 0]
    idx = [int(np.argmin(np.abs(ts - f / fps))) for f in range(len(REF))]
    out = []
    for n in NAMES:
        i = names.index(n); T4 = np.tile(np.eye(4), (len(idx), 1, 1)); T4[:, :3, 3] = P[idx, i]; T4[:, :3, :3] = R.from_quat(Q[idx, i][:, [1, 2, 3, 0]]).as_matrix(); out.append(T4)
    return np.stack(out, 1)

if __name__ == '__main__':
    X = load_traj(sys.argv[1]) if sys.argv[1].endswith('.npz') else load_results(sys.argv[1]); o, pos, ori = score(X, True); print(json.dumps(o))
    for s in range(0, len(pos), 20): print('t=%4.1fs cup %.2f cm %.1f deg | pot %.2f cm %.1f deg' % (s / 20, pos[s:s+20, 0].mean(), ori[s:s+20, 0].mean(), pos[s:s+20, 1].mean(), ori[s:s+20, 1].mean()))
