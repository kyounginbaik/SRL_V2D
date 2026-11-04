"""Scoring-only: split of the frame-0 scene-alignment error into translation and rotation parts. Usage EP TAG"""
import sys, json, numpy as np, kitx
from scipy.spatial.transform import Rotation as Rot
ep, tag = int(sys.argv[1]), sys.argv[2]; X = kitx.load_export(ep, tag)
o = kitx.score(ep, {n: X['W'][:, i] for i, n in enumerate(X['names'])}, X['meshes'])
M = kitx.M; A, R = o['A'], o['R']; out = {}
P, Q = [], []
for s in range(len(o['objs'])):
    Ra, Rr = M._rotation_matrix(A[:, s, 3:]), M._rotation_matrix(R[:, s, 3:]); ax = o['axes'][s]
    ks = [np.zeros(3)] + ([ax * .05, -ax * .05] if o['orders'][s] == 360 else [] if np.any(ax) else [np.eye(3)[i] * .05 * sg for i in range(3) for sg in (1, -1)])
    for k in ks: P.append(A[:, s, :3] + Ra @ k); Q.append(R[:, s, :3] + Rr @ k)
P, Q = np.concatenate(P), np.concatenate(Q)
t_only = (Q - P).mean(0)
p0 = A[0, 0, :3]  # rotation about slot-0 f0 point (which is pinned)
U, _, Vt = np.linalg.svd((P - p0).T @ (Q - p0)); d = np.sign(np.linalg.det(Vt.T @ U.T)); S = Vt.T @ np.diag([1, 1, d]) @ U.T
for s, n in enumerate(o['objs']):
    e0 = np.linalg.norm(A[:, s, :3] - R[:, s, :3], axis=1).mean() * 100
    et = np.linalg.norm(A[:, s, :3] + t_only - R[:, s, :3], axis=1).mean() * 100
    er = np.linalg.norm((A[:, s, :3] - p0) @ S.T + p0 - R[:, s, :3], axis=1).mean() * 100
    out[n] = (e0, et, er)
    print(f"{ep:3d} {n:18s} f0-align {e0:.2f} | +best translation {et:.2f} | +best rotation about slot0 {er:.2f} | rot {np.degrees(Rot.from_matrix(S).magnitude()):.2f} deg trans {np.linalg.norm(t_only)*100:.2f} cm")
json.dump(out, open(f'../out/orsplit_{tag}_{ep}.json', 'w'))
