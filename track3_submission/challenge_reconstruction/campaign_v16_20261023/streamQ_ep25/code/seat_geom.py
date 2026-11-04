"""Insertion geometry of an executed trajectory: board-2 slot centre / plate vs board 1 (PCA, cm), angle between long axes, mid-axis
tilt, heights. usage: seat_geom.py TRAJ.npz [STRIDE]"""
import sys, numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, str(Path(__file__).resolve().parent)); import correct_ep25_v2 as C
F = C.frames(Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/collision_recon_slots_v3/episode_000025')); f1, f2 = F['wooden_piece_1'], F['wooden_piece_2']
tr = np.load(sys.argv[1]); st = int(sys.argv[2]) if len(sys.argv) > 2 else 10
def T(k, j): M = np.eye(4); M[:3, :3] = R.from_quat(tr['object_quaternions_wxyz'][k, j][[1, 2, 3, 0]]).as_matrix(); M[:3, 3] = tr['object_positions'][k, j]; return M
for k in range(200, len(tr['object_positions']), st):
    T1, T2 = T(k, 0), T(k, 1); s2 = f2['c0'] + f2['U'] @ np.array([f2['plate_thin'], 0, f2['slot_long']]); p = T2[:3, :3] @ s2 + T2[:3, 3]
    q = f1['U'].T @ (T1[:3, :3].T @ (p - T1[:3, 3]) - f1['c0']); c2 = T2[:3, :3] @ f2['c0'] + T2[:3, 3]; qc = f1['U'].T @ (T1[:3, :3].T @ (c2 - T1[:3, 3]) - f1['c0'])
    M = f1['U'].T @ T1[:3, :3].T @ T2[:3, :3] @ f2['U']
    print(f"{k/20:5.2f} thin {100*(q[0]-f1['plate_thin']):5.1f} long {100*(qc[2]-f1['slot_long']):5.1f} mid {100*qc[1]:5.1f} | longang {np.degrees(np.arccos(abs(M[2,2]))):4.0f} midang {np.degrees(np.arccos(min(1,abs(M[1,1])))):4.0f} | z1 {T1[2,3]*100:4.1f} z2 {T2[2,3]*100:4.1f}")
