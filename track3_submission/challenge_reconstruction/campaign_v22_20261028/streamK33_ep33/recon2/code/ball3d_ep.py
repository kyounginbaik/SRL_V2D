"""K33-recon2 copy of ball3d33.py with the episode as argv[2]. K33-inhand (GT-free): per-frame 3D residual between OUR stereo ball detections (camera_point) and our ball model (G18 pooled
centres near/far/tip) placed by a bundle's camera_from_object for the spoon; plus a ball-only pose (Kabsch over all assignments) on
frames with >=3 detections, and its rotation/roll vs the bundle. usage: ball3d33.py BUNDLE_EP_DIR [out.txt]"""
import sys, json, itertools, numpy as np
from scipy.spatial.transform import Rotation as Rot
B = sys.argv[1]; EPN = int(sys.argv[2]); sys.argv = [sys.argv[0], B] + sys.argv[3:]
C = f'/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/canonical/evaluation/episode_{EPN:06d}'
z = np.load(f'{B}/object_poses.npz', allow_pickle=False); S = list(z['names']).index('wooden_spoon'); Tc = z['camera_from_object'][:, S]
obs = z['object_observed'][:, S]
bc = json.load(open('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/streamG18_spoonmesh/out/ballcenters.json'))
pe = bc['per_episode_centres']; M = np.array([np.median([pe[e][k] for e in pe], 0) for k in ('near', 'far', 'tip')])
fr = json.load(open(f'{C}/ball_detections_wooden_spoon.json'))['frames']
long = M[2] - M[0]; long /= np.linalg.norm(long)
def kabsch(A, Bp):
    a, b = A.mean(0), Bp.mean(0); H = (A - a).T @ (Bp - b); U, s, Vt = np.linalg.svd(H); D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T; return R, b - R @ a
out = []
for t in range(len(Tc)):
    D = np.array([d['camera_point'] for d in fr[t]]) if fr[t] else np.zeros((0, 3))
    P = M @ Tc[t][:3, :3].T + Tc[t][:3, 3]
    res = np.linalg.norm(D[:, None] - P[None], axis=2).min(1) * 100 if len(D) else np.array([])
    fit = ''
    if len(D) >= 3:
        best = None
        for idx in itertools.permutations(range(len(D)), 3):
            R, tt = kabsch(M, D[list(idx)]); e = np.linalg.norm(M @ R.T + tt - D[list(idx)], axis=1).max() * 100
            if best is None or e < best[0]: best = (e, R, tt)
        e, R, tt = best
        if e < 1.0:
            dR = Tc[t][:3, :3].T @ R; rv = Rot.from_matrix(dR).as_rotvec(); ang = np.degrees(np.linalg.norm(rv))
            roll = np.degrees(rv @ long); dp = np.linalg.norm(tt - Tc[t][:3, 3] + (R - Tc[t][:3, :3]) @ M.mean(0) * 0) * 100
            fit = f' ballpose fit {e:.2f}cm  dRot {ang:5.1f}d  roll-comp {roll:6.1f}d'
    out.append(f'f{t:3d} obs={int(obs[t])} ndet={len(D)} res_cm=' + (' '.join(f'{r:5.1f}' for r in res) if len(res) else '-') + fit)
txt = '\n'.join(out); print(txt)
if len(sys.argv) > 2: open(sys.argv[2], 'w').write(txt + '\n')
