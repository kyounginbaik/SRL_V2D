"""Stream P, episode 33: spoon roll-branch correction in frames ~134-165 (phase-5 h2h finding; our track is the 180-deg roll
twin at f143-154 by the mocap balls).

Method (keeps OUR camera chain, our position track and our mesh frame): R_new(t) = R_ours(t) * Rx(phi(t)), a roll about the spoon
body x (long) axis through the body origin; positions unchanged. phi(t) is the 1-dof roll that best explains the mocap balls
(code/roll_scan_balls.py -> roll/ball_roll_scan_110_180.json: canonical cross-episode ball geometry, our camera model), taken at
these knots and interpolated monotonically (PCHIP):
   f134: 0 (balls: f131-133 best phi 0..+10, phi0 76 px vs twin 108 px)  -- blend-in start, 0.5 s ramp, no balls in f134-139
   f144: -175, f147: -170, f150: -150, f153: -140, f156: -120, f159: -70, f162: -40, f165: 0  (ball scan best phi per 3 frames)
The coworker's NEW track rolls in the same direction (-100..-125 deg vs ours in f134-158), confirming the sign of the path.
Writes (new dirs only): staging/source_corrected/episode_000033 (copy of final_recon/episode_000033 with object_poses.npz and
wooden_spoon_trajectory.npz rewritten + roll_correction.json) and staging/episode_corrected/episode_000033 (copy of the legacy
episode dir with tracking_reference.npz / reference_trajectory.npz spoon quaternions replaced and input_manifest.json pointing at
the corrected source, same convention as refinement_20261021_ep36/episode_corrected). usage: correct_ep33_roll.py"""
import json, shutil, hashlib, numpy as np
from pathlib import Path
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation as R
P = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamP_ep33')
SRC = Path('/data1/SRL_V2D/challenge_reconstruction/final_recon/episode_000033')
EPD = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/episodes/episode_000033')
OUT, OEP = P / 'staging/source_corrected/episode_000033', P / 'staging/episode_corrected/episode_000033'
KNOTS = [(134, 0.), (144, -175.), (147, -170.), (150, -150.), (153, -140.), (156, -120.), (159, -70.), (162, -40.), (165, 0.)]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
if OUT.exists() or OEP.exists(): raise SystemExit('exists')
z = dict(np.load(SRC / 'object_poses.npz', allow_pickle=False)); names = [str(x) for x in z['names']]; si = names.index('wooden_spoon')
T = z['world_from_object'].copy(); n = len(T); f = np.arange(n)
kf, kp = np.array([k[0] for k in KNOTS]), np.array([k[1] for k in KNOTS])
phi = np.zeros(n); m = (f >= kf[0]) & (f <= kf[-1]); phi[m] = PchipInterpolator(kf, kp)(f[m])
for k in np.flatnonzero(phi != 0): T[k, si, :3, :3] = T[k, si, :3, :3] @ R.from_euler('x', phi[k], degrees=True).as_matrix()
shutil.copytree(SRC, OUT, symlinks=True); z['world_from_object'] = T; np.savez_compressed(OUT / 'object_poses.npz', **z)
if (OUT / 'wooden_spoon_trajectory.npz').exists():
    tr = dict(np.load(SRC / 'wooden_spoon_trajectory.npz', allow_pickle=False)); changed = []
    for key in list(tr):
        a = tr[key]
        if a.ndim == 3 and a.shape[1:] == (4, 4) and len(a) == n:
            new = a.copy()
            for k in np.flatnonzero(phi != 0): new[k, :3, :3] = a[k, :3, :3] @ R.from_euler('x', phi[k], degrees=True).as_matrix()
            tr[key] = new; changed.append(key)
    np.savez_compressed(OUT / 'wooden_spoon_trajectory.npz', **tr)
else: changed = []
# legacy episode dir copy with the setup-frame spoon quaternions replaced
shutil.copytree(EPD, OEP, symlinks=True); man = json.loads((EPD / 'input_manifest.json').read_text()); S = np.asarray(man['setup_from_source'])
qs = R.from_matrix((S @ T[:, si])[:, :3, :3]).as_quat(scalar_first=True)
t = dict(np.load(EPD / 'tracking_reference.npz', allow_pickle=False)); jn = [str(x) for x in t['joint_names']]; c = jn.index('wooden_spoon_free[3]')
old = t['qpos'][:, c:c + 4]; qs = qs * np.sign((qs * old).sum(1, keepdims=True)); t['qpos'] = t['qpos'].copy(); t['qpos'][:, c:c + 4] = qs
np.savez_compressed(OEP / 'tracking_reference.npz', **t)
rt = dict(np.load(EPD / 'reference_trajectory.npz', allow_pickle=False)); rt['qpos'] = rt['qpos'].copy(); rt['qpos'][:, c:c + 4] = qs
oq = rt['object_quaternions_wxyz'].copy(); oq[:, si] = qs; rt['object_quaternions_wxyz'] = oq; np.savez_compressed(OEP / 'reference_trajectory.npz', **rt)
scan = json.load(open(P / 'roll/ball_roll_scan_110_180.json'))
rec = dict(episode=33, object='wooden_spoon', method='roll about the local long X axis through the body origin, R_new = R_ours @ Rx(phi); positions unchanged; phi = PCHIP through ball-scan knots',
    knots_frame_deg=KNOTS, frames_changed=int((phi != 0).sum()), window_frames=[int(kf[0]), int(kf[-1])], window_s=[kf[0] / 20, kf[-1] / 20], max_phi_deg=float(np.abs(phi).max()),
    phi_deg_per_frame={int(k): round(float(phi[k]), 2) for k in np.flatnonzero(phi != 0)},
    evidence=dict(ball_scan=str(P / 'roll/ball_roll_scan_110_180.json'), ball_scan_sha256=sha(P / 'roll/ball_roll_scan_110_180.json'),
        ball_rows=[dict(frame=r['frame'], ours_px=round(r['ours'], 1), twin_px=round(r['twin'], 1), new_px=round(r['new'], 1), best_phi=scan['phis'][int(np.argmin(r['res_by_phi']))], best_px=round(min(r['res_by_phi']), 1)) for r in scan['rows']],
        h2h=['/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/h2h_spoon/README.md', '/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/h2h_spoon/verdicts.json', '/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/h2h_spoon/figures/zoom33_roll.jpg'],
        depth_iou=str(P / 'roll/evidence_roll.json'), audit=str(P / 'audit/roll_corrected.txt'),
        new_track_relative_roll='coworker NEW track (own camera model, registered mesh frame) vs ours: body roll -100..-125 deg in f134-158, -67..-45 deg in f161-173 (same direction)'),
    trajectory_keys_rewritten=changed, original_object_poses_sha256=sha(SRC / 'object_poses.npz'), corrected_object_poses_sha256=sha(OUT / 'object_poses.npz'), script=str(Path(__file__).resolve()), script_sha256=sha(__file__))
(OUT / 'roll_correction.json').write_text(json.dumps(rec, indent=1))
man['source'] = str(OUT); man['source_reference'] = str(OUT / 'reference.parquet'); man['source_reference_sha256'] = sha(OUT / 'reference.parquet'); man['source_object_poses_sha256'] = sha(OUT / 'object_poses.npz')
man['corrected_track_note'] = 'wooden_spoon roll branch corrected on f134-165 (6.70-8.25 s), see source_corrected/episode_000033/roll_correction.json; robot/scene unchanged'
man['roll_correction'] = dict(record=str(OUT / 'roll_correction.json'), record_sha256=sha(OUT / 'roll_correction.json'), original_source=str(SRC), original_source_object_poses_sha256=sha(SRC / 'object_poses.npz'), original_manifest=str(EPD / 'input_manifest.json'), original_manifest_sha256=sha(EPD / 'input_manifest.json'))
(OEP / 'input_manifest.json').write_text(json.dumps(man, indent=2) + '\n')
print(json.dumps({k: rec[k] for k in ('frames_changed', 'window_s', 'max_phi_deg', 'corrected_object_poses_sha256', 'trajectory_keys_rewritten')}))
