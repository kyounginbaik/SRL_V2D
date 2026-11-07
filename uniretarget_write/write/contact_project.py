"""Contact-consistent reference, in two cheap steps:
  1. One constant grip offset (hand relative to the object, 6 numbers): the smallest one for which the hand with straight
     fingers does not pass through the object on any sampled frame (random search, then local refinement).
  2. Per frame, each finger closes toward its IK pose but stops at the object surface.
The wrist keeps the human motion (only the constant offset is added). Usage: contact_project.py OUT.npz [TOL_MM]"""
import sys, numpy as np, mujoco, json
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, '/scratch2/exaFLOPs/srl_yam_run/ablate/oak/write'); import wmetric as W
from planner import base_from, wrist_T, pen_T
out = sys.argv[1]; TOL = float(sys.argv[2]) / 1000 if len(sys.argv) > 2 else 0.0005
m = mujoco.MjModel.from_xml_path(str(W.P / 'scene.xml')); d = mujoco.MjData(m); b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'right_hand_C_MC')
gn = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
jn = lambda j: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ''
PG = [g for g in range(m.ngeom) if gn(g).startswith('right_object_') and 'visual' not in gn(g)]
FINGERS = ('thumb', 'index', 'middle', 'ring', 'pinky')
geoms = {f: [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_group[g] == 1 and f'right_{f}' in gn(g)] for f in FINGERS}
flex = {f: [m.jnt_qposadr[j] for j in range(m.njnt) if f'right_{f}' in jn(j) and any(s in jn(j) for s in ('FE', 'PIP', 'DIP', 'IP'))] for f in FINGERS}
palm = [g for g in range(m.ngeom) if 'C_MC' in gn(g) and m.geom_group[g] == 1 and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
HAND = sum(geoms.values(), []) + palm
ALLFLEX = np.concatenate(list(flex.values()))
fr = np.zeros(6); pid = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'right_object_joint')]
def depth(q, gs):
    d.qpos[:] = q; mujoco.mj_forward(m, d)
    return max(0.0, -min(mujoco.mj_geomDistance(m, d, h, p, .03, fr) for h in gs for p in PG))
Q = W.RF['q'].copy(); n = len(Q)
# hand pose in the object frame, per frame
Xo = [pen_T(Q[i, pid:pid + 7]) for i in range(n)]
Xh = [wrist_T(m, d, b, Q[i]) for i in range(n)]
def with_offset(i, dp, dr, q=None):
    """Hand moved by a constant offset expressed in the object frame."""
    O = np.eye(4); O[:3, :3] = R.from_rotvec(dr).as_matrix(); O[:3, 3] = dp
    rel = np.linalg.inv(Xo[i]) @ Xh[i]
    Xw = Xo[i] @ O @ rel
    q = Q[i].copy() if q is None else q.copy()
    q[:6] = base_from(m, d, b, Xw, q)
    return q
S = list(range(0, n, max(1, n // 16)))
def cost(dp, dr):
    worst = 0.0
    for i in S:
        q = with_offset(i, dp, dr); q[ALLFLEX] = 0.0
        worst = max(worst, depth(q, HAND))
        if worst > 0.03: break
    return worst
rng = np.random.default_rng(0); best = None
for trial in range(300):
    scale = min(1.0, 0.2 + trial / 150)
    dp, dr = rng.uniform(-0.025, 0.025, 3) * scale, rng.uniform(-np.radians(20), np.radians(20), 3) * scale
    c = cost(dp, dr)
    if c <= TOL:
        size = np.linalg.norm(dp) / 0.01 + np.linalg.norm(dr) / np.radians(10)
        if best is None or size < best[0]: best = (size, dp, dr)
if best is None: sys.exit('no offset clears the object; widen the search')
# shrink toward zero while it still clears the object
size, dp, dr = best
for _ in range(12):
    a = 0.9
    if cost(dp * a, dr * a) <= TOL: dp, dr = dp * a, dr * a
    else: break
P = np.zeros_like(Q)
for i in range(n):
    q = with_offset(i, dp, dr)
    for f in FINGERS:
        target = Q[i, flex[f]].copy(); q[flex[f]] = target
        if depth(q, geoms[f]) <= TOL: continue
        lo, hi = 0.0, 1.0
        for _ in range(10):
            mid = (lo + hi) / 2; q[flex[f]] = target * mid
            if depth(q, geoms[f]) <= TOL: lo = mid
            else: hi = mid
        q[flex[f]] = target * lo
    P[i] = q
P[:, 6:28] = gaussian_filter1d(P[:, 6:28], 1.0, axis=0, mode='nearest')
ov_ik = [depth(Q[i], HAND) for i in range(0, n, 10)]; ov_p = [depth(P[i], HAND) for i in range(0, n, 10)]
np.savez(out, q=P, offset_pos=dp, offset_rotvec=dr)
res = dict(frames=n, offset_cm=np.round(dp * 100, 2).tolist(), offset_deg=np.round(np.degrees(dr), 1).tolist(), overlap_ik_med_cm=round(float(np.median(ov_ik)) * 100, 2),
           overlap_proj_med_cm=round(float(np.median(ov_p)) * 100, 2), overlap_proj_max_cm=round(float(np.max(ov_p)) * 100, 2),
           finger_open_rad_med=round(float(np.median(np.abs(P[:, 6:28] - Q[:, 6:28]).max(1))), 3))
json.dump(res, open(out.replace('.npz', '.json'), 'w')); print(res)
