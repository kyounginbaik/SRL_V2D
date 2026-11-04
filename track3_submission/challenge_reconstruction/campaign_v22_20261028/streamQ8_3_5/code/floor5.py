"""Q8 ep5: floor analysis of the spoon position error of an actor rollout vs target_final_v5 (scorer-aligned frames).
usage: floor5.py EP ACTOR_DIR MESHDIR [OUT.json]
Reports: target jitter (|ref - G_s(ref)|), achieved-trajectory roughness, error spectrum split (low-pass part removable by an ideal
low-frequency correction vs residual high-frequency part), and windowed errors."""
import sys, json, numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter1d as gf
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamL8_3_5_6/code')
import plib as B
ep = int(sys.argv[1]); actor = Path(sys.argv[2]); meshdir = Path(sys.argv[3])
reg = B.register(ep, {n: meshdir / f'{n}.obj' for n in B.NAMES})
import mujoco
from scipy.spatial.transform import Rotation as R
ev = json.loads((actor / 'evaluation/result.json').read_text()); m = mujoco.MjModel.from_xml_path(ev['execution']['scene'])
Q = np.load(actor / 'evaluation/execution/trajectory.npz')['qpos']
P = {}
for n in B.NAMES:
    a = int(m.joint(n + '_free').qposadr[0]); T = np.tile(np.eye(4), (len(Q), 1, 1))
    T[:, :3, :3] = R.from_quat(Q[:, a + 3:a + 7], scalar_first=True).as_matrix(); T[:, :3, 3] = Q[:, a:a + 3]; P[n] = T
e, ach, ref = B.per_frame_full(ep, P, reg)
out = {}
for s, n in enumerate(B.NAMES):
    pa = ach[:, s, :3] * 100; pr = ref[:, s, :3] * 100; ev_ = pa - pr; N = len(pa)
    o = dict(mean=float(np.linalg.norm(ev_, axis=1).mean()), check_vs_per_frame=float(e[n][0].mean()), bias_cm=ev_.mean(0).round(3).tolist())
    o['target_jitter'] = {str(sg): float(np.linalg.norm(pr - gf(pr, sg, axis=0, mode='nearest'), axis=1).mean()) for sg in (0.5, 1, 1.5, 2, 3, 4, 6)}
    o['achieved_roughness'] = {str(sg): float(np.linalg.norm(pa - gf(pa, sg, axis=0, mode='nearest'), axis=1).mean()) for sg in (0.5, 1, 1.5, 2, 3, 4, 6)}
    # ideal removal of the low-frequency error component (cutoff sigma frames): residual = e - G_s(e)
    o['err_after_lowpass_removed'] = {str(sg): float(np.linalg.norm(ev_ - gf(ev_, sg, axis=0, mode='nearest'), axis=1).mean()) for sg in (1, 2, 3, 4, 6, 10, 20)}
    o['err_lowpass_part'] = {str(sg): float(np.linalg.norm(gf(ev_, sg, axis=0, mode='nearest'), axis=1).mean()) for sg in (1, 2, 3, 4, 6, 10, 20)}
    # if the spoon followed the s-smoothed target exactly AND kept its own present high-freq motion? -> report both bounds
    o['follow_smoothed_target'] = o['target_jitter']
    w = 10; o['windows_0.5s'] = [round(float(np.linalg.norm(ev_[k:k + w], axis=1).mean()), 2) for k in range(0, N, w)]
    o['target_speed_cm_s_p50_p95'] = np.percentile(np.linalg.norm(np.diff(pr, axis=0), axis=1) * 20, [50, 95]).round(2).tolist()
    o['ach_speed_cm_s_p50_p95'] = np.percentile(np.linalg.norm(np.diff(pa, axis=0), axis=1) * 20, [50, 95]).round(2).tolist()
    out[n] = o
print(json.dumps(out, indent=1))
if len(sys.argv) > 4: Path(sys.argv[4]).write_text(json.dumps(out, indent=1))
np.savez_compressed(Path(sys.argv[4]).with_suffix('.npz') if len(sys.argv) > 4 else '/dev/null', ach=ach, ref=ref)
# orientation analysis for the spoon (slot 1)
from scipy.spatial.transform import Rotation as Rr_
s = 1; qa = ach[:, s, 3:].copy(); qr = ref[:, s, 3:].copy()
for q in (qa, qr):
    for k in range(1, len(q)):
        if (q[k] * q[k - 1]).sum() < 0: q[k] *= -1
def ang(q1, q2): return np.degrees((Rr_.from_quat(q1).inv() * Rr_.from_quat(q2)).magnitude())
def smq(q, sg): x = gf(q, sg, axis=0, mode='nearest'); return x / np.linalg.norm(x, axis=1, keepdims=True)
D = (Rr_.from_quat(qr).inv() * Rr_.from_quat(qa)).as_rotvec()
oo = dict(mean=float(ang(qa, qr).mean()),
          target_jitter_deg={str(sg): float(ang(qr, smq(qr, sg)).mean()) for sg in (1, 2, 3, 4, 6)},
          err_after_lowpass_removed_deg={str(sg): float(np.degrees(np.linalg.norm(D - gf(D, sg, axis=0, mode='nearest'), axis=1)).mean()) for sg in (1, 2, 3, 4, 6, 10)},
          windows_05=None)
oo["windows_05"] = [round(float(ang(qa, qr)[k:k + 10].mean()), 1) for k in range(0, len(qa), 10)]
print('SPOON ORIENTATION', json.dumps(oo))
