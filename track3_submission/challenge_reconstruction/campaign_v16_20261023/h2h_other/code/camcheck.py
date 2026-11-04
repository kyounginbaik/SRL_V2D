"""Verify that OUR camera (world_from_rgb) and the NEW camera (cam0_T_cam) are the same rig motion seen through
A_T_rect: relative motions A_o(t) = inv(Wc0) Wc_t vs X B_n(t) X^-1, X = A_T_rect (calibration). Also solve X freely
(rotation by Park-Martin on relative motions) and report its angle to the calibration X, and the world-to-world map."""
import sys, json, numpy as np
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, __import__('os').path.dirname(__file__)); from common import *
res = {}
for ep in [1, 8, 10, 14, 19, 20, 25, 28, 30, 38]:
    o = load_ours(ep); n = load_new(ep, o['names']); Wc = o['world_from_cam']; C = n['cam0_T_cam']; X = A_T_rect(ep)
    T = min(len(Wc), len(C)); A = np.einsum('ij,tjk->tik', np.linalg.inv(Wc[0]), Wc[:T]); B = np.einsum('ij,tjk->tik', np.linalg.inv(C[0]), C[:T])
    pred = np.einsum('ij,tjk,kl->til', X, B, np.linalg.inv(X)); dR = [np.degrees(np.linalg.norm(R.from_matrix(A[t][:3, :3].T @ pred[t][:3, :3]).as_rotvec())) for t in range(T)]
    dt = np.linalg.norm(A[:, :3, 3] - pred[:, :3, 3], axis=1) * 100
    # free hand-eye rotation
    M = np.zeros((3, 3))
    for t in range(1, T, 2):
        a = R.from_matrix(A[t][:3, :3]).as_rotvec(); b = R.from_matrix(B[t][:3, :3]).as_rotvec()
        if np.linalg.norm(a) > np.radians(2): M += np.outer(b, a)
    U, S, Vt = np.linalg.svd(M); Rx = Vt.T @ U.T
    if np.linalg.det(Rx) < 0: Vt[-1] *= -1; Rx = Vt.T @ U.T
    ang = np.degrees(np.linalg.norm(R.from_matrix(Rx.T @ X[:3, :3]).as_rotvec()))
    res[ep] = dict(frames_ours=len(Wc), frames_new=len(C), rel_rot_err_deg_med=float(np.median(dR)), rel_rot_err_deg_max=float(np.max(dR)),
                   rel_trans_err_cm_med=float(np.median(dt)), rel_trans_err_cm_max=float(np.max(dt)), free_handeye_vs_calib_deg=float(ang),
                   cam_path_len_cm=float(np.sum(np.linalg.norm(np.diff(Wc[:, :3, 3], axis=0), axis=1)) * 100))
    print(ep, json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in res[ep].items()}), flush=True)
json.dump(res, open(sys.argv[1], 'w'), indent=1)
