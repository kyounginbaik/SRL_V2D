"""AX32 'K32_SM': despike the ep32 spoon track of the K32_R10 bundle in the blurred fast-motion bursts (GT-free, image-consistent).
Bursts (frames of tracker jitter / bad mask fit): given on the command line as a,b ranges. Inside each burst the spoon orientation is replaced by
SLERP between the burst boundary frames (a-1, b+1) and the position by a cubic Hermite blend using the good neighbours; then a rotation-vector
Savitzky-Golay low-pass (window 9, order 2) is applied over the burst +-4 frames; finally an angular-velocity cap of CAP deg/frame is enforced
by iterative SLERP smoothing over the whole track except frames 0..34 (the R10 roll fix, kept byte-identical).
usage: build_k32sm.py LABEL CAP_DEG a1,b1 a2,b2 ..."""
import sys, json, shutil, hashlib, numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as Rot, Slerp
from scipy.signal import savgol_filter
from scipy.interpolate import CubicSpline
K = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK32_ep32'); SRC = K / 'bundles/K32_R10/episode_000032'
lab, cap = sys.argv[1], float(sys.argv[2]); bursts = [tuple(int(x) for x in a.split(',')) for a in sys.argv[3:]]
out = K / 'bundles' / lab / 'episode_000032'; assert not out.exists(); shutil.copytree(SRC, out)
z = np.load(SRC / 'object_poses.npz', allow_pickle=False); d = {k: z[k] for k in z.files}; KS = list(d['names']).index('wooden_spoon')
W = d['world_from_object'][:, KS].copy(); n = len(W); KEEP = 35
R = Rot.from_matrix(W[:, :3, :3]); P = W[:, :3, 3].copy()
def speed(Rr): return np.array([np.degrees((Rr[f + 1] * Rr[f].inv()).magnitude()) for f in range(len(Rr) - 1)])
sp0 = speed(R)
q = R.as_quat(); p = P.copy()
for a, b in bursts:
    a = max(a, KEEP); lo, hi = a - 1, min(b + 1, n - 1)
    if lo < KEEP or hi <= lo: continue
    s = Slerp([lo, hi], Rot.from_quat(q[[lo, hi]])); t = np.arange(lo, hi + 1); q[lo:hi + 1] = s(t).as_quat()
    g = np.r_[np.arange(max(lo - 4, KEEP), lo + 1), np.arange(hi, min(hi + 5, n))]      # good neighbours
    cs = CubicSpline(g, p[g], axis=0); p[lo:hi + 1] = cs(t)
    w0, w1 = max(lo - 4, KEEP), min(hi + 4, n - 1); rv = Rot.from_quat(q[w0:w1 + 1]).as_rotvec()
    if len(rv) >= 9: q[w0:w1 + 1] = Rot.from_rotvec(savgol_filter(rv, 9, 2, axis=0)).as_quat()
# angular-velocity cap by iterative SLERP smoothing (frames >= KEEP only)
for it in range(200):
    Rr = Rot.from_quat(q); s = speed(Rr); bad = np.flatnonzero(s[KEEP:] > cap) + KEEP
    if len(bad) == 0: break
    for f in bad:
        f0, f1 = max(f - 1, KEEP), min(f + 2, n - 1)
        if f1 - f0 < 2: continue
        sl = Slerp([f0, f1], Rot.from_quat(q[[f0, f1]])); mid = np.arange(f0 + 1, f1); q[mid] = sl(mid).as_quat()
Rn = Rot.from_quat(q); W2 = W.copy(); W2[:, :3, :3] = Rn.as_matrix(); W2[:, :3, 3] = p
assert np.allclose(W2[:KEEP], W[:KEEP])
d['world_from_object'][:, KS] = W2
for t in range(n): d['camera_from_object'][t, KS] = np.linalg.inv(d['world_from_rgb'][t]) @ W2[t]
(out / 'object_poses.npz').unlink(); np.savez_compressed(out / 'object_poses.npz', **d)
sp1 = speed(Rn); dpos = np.linalg.norm(W2[:, :3, 3] - W[:, :3, 3], axis=1) * 100; dori = np.degrees((Rn * R.inv()).magnitude())
rec = dict(stream='K32-AX32', label=lab, source=str(SRC), ground_truth_used=False, rule='despike/smooth spoon track in blurred bursts + angular velocity cap', cap_deg_per_frame=cap, bursts=bursts, keep_frames=KEEP,
           max_rot_per_frame_before=float(sp0.max()), max_rot_per_frame_after=float(sp1.max()), n_frames_over_15_before=int((sp0 > 15).sum()), n_frames_over_15_after=int((sp1 > 15).sum()),
           max_pos_change_cm=float(dpos.max()), mean_pos_change_cm=float(dpos.mean()), max_ori_change_deg=float(dori.max()), mean_ori_change_deg=float(dori.mean()),
           object_poses_sha256=hashlib.sha256((out / 'object_poses.npz').read_bytes()).hexdigest())
(out / 'K32SM_CHANGES.json').write_text(json.dumps(rec, indent=1))
tg = K / 'targets' / lab / 'episode_000032'; tg.mkdir(parents=True)
np.savez_compressed(tg / 'wooden_spoon_trajectory.npz', T_world_obj=W2)
V14T = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/target_final_v14_g17/episode_000032')
for fn in ('blue_cup_trajectory.npz', 'blue_cup_mesh_metric.obj'): shutil.copy2(V14T / fn, tg / fn)
shutil.copy2(out / 'meshes' / 'wooden_spoon.obj', tg / 'wooden_spoon_mesh_metric.obj')
print(lab, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in rec.items() if k not in ('source', 'object_poses_sha256', 'rule')})
print('per-frame rot before by 10:', ' '.join(f'{sp0[f:f+10].max():3.0f}' for f in range(0, n - 1, 10)))
print('per-frame rot after  by 10:', ' '.join(f'{sp1[f:f+10].max():3.0f}' for f in range(0, n - 1, 10)))
