"""Stream R ep32 phase-5 correction: spoon roll from the coworker's NEW track in the windows where the h2h mocap balls favour NEW.
R_corr(t) = R_ours(t) * Rx(w(t) * phi_new(t)): phi_new = NEW's roll about OUR spoon long body axis, measured in OUR camera frame
(roll/roll_balls.json from code/roll_balls.py: NEW pose -> our Kout camera via the h2h best-fit rotation, NEW body -> our body via the
h2h mesh registration G); w = smoothstep ramps centred on the window edges. Positions, the other two rotation components, our camera
chain and the cup are unchanged (the NEW-vs-ours difference is a roll of 148-178 deg about +-x with 1-8 deg off-axis; balls do not
separate NEW's roll from our pure half-turn twin, so NEW's measured roll is used as instructed). Writes:
  OUT/source_corrected/episode_000032   (copy of final_recon/episode_000032 with object_poses.npz + wooden_spoon_trajectory.npz rewritten, roll_correction.json)
  OUT/episode_corrected/episode_000032  (copy of the shipped repaired episode dir; input_manifest -> corrected source; tracking_reference.npz /
                                         reference_trajectory.npz spoon object columns = setup_from_source @ corrected poses)
usage: correct_ep32_roll.py OUT_DIR RAMP_S F0-F1 [F0-F1 ...]"""
import sys, json, shutil, hashlib, numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
S = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamR_ep32')
SRC = Path('/data1/SRL_V2D/challenge_reconstruction/final_recon/episode_000032')
EPD = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/expanded_batch/repaired_export_inputs/episode_000032')
out = Path(sys.argv[1]).resolve(); ramp = float(sys.argv[2]); wins = [tuple(map(int, w.split('-'))) for w in sys.argv[3:]]
OS, OE = out / 'source_corrected/episode_000032', out / 'episode_corrected/episode_000032'
if OS.exists() or OE.exists(): raise SystemExit('exists ' + str(out))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
rb = json.load(open(S / 'roll/roll_balls.json')); rows = rb['rows']; n = len(rows); f = np.arange(n, dtype=float)
phi_raw = np.array([r['phi_new_deg'] for r in rows])
sm = lambda u: np.clip(u, 0, 1) ** 3 * (10 - 15 * np.clip(u, 0, 1) + 6 * np.clip(u, 0, 1) ** 2)
h = ramp * 20 / 2   # half ramp in frames
phi = np.zeros(n); wsum = np.zeros(n); per_window = []
for f0, f1 in wins:
    lo, hi = max(0, int(np.floor(f0 - h - 1))), min(n - 1, int(np.ceil(f1 + h + 1)))
    u = np.degrees(np.unwrap(np.radians(phi_raw[lo:hi + 1]))); u -= 360 * np.round(np.mean(u[f0 - lo:f1 - lo + 1]) / 360)   # window mean in (-180, 180]
    w = sm((f[lo:hi + 1] - (f0 - .5 - h)) / (2 * h)) - sm((f[lo:hi + 1] - (f1 + .5 - h)) / (2 * h))
    if np.any(wsum[lo:hi + 1][w > 1e-9] > 1e-9): raise SystemExit('ramps overlap')
    phi[lo:hi + 1] += w * u; wsum[lo:hi + 1] += w; per_window.append(dict(frames=[f0, f1], mean_roll_deg=float(np.mean(u[f0 - lo:f1 - lo + 1])), ramp_frames=[lo, hi]))
shutil.copytree(SRC, OS, symlinks=True)
z = dict(np.load(SRC / 'object_poses.npz', allow_pickle=False)); names = [str(x) for x in z['names']]; si = names.index('wooden_spoon')
T = z['world_from_object'].copy(); T0 = T.copy()
for k in range(n):
    if phi[k] != 0: T[k, si, :3, :3] = T[k, si, :3, :3] @ R.from_euler('x', phi[k], degrees=True).as_matrix()
z['world_from_object'] = T; (OS / 'object_poses.npz').unlink(); np.savez_compressed(OS / 'object_poses.npz', **z)
if (OS / 'wooden_spoon_trajectory.npz').exists():
    tr = dict(np.load(SRC / 'wooden_spoon_trajectory.npz', allow_pickle=False)); changed = []
    for key in list(tr):
        a = tr[key]
        if a.ndim == 3 and a.shape[-2:] == (4, 4) and len(a) == n:
            if np.allclose(a, T0[:, si], atol=1e-9): tr[key] = T[:, si]; changed.append(key)
            elif np.allclose(a, np.linalg.inv(z['world_from_rgb']) @ T0[:, si], atol=1e-9): tr[key] = np.linalg.inv(z['world_from_rgb']) @ T[:, si]; changed.append(key)
    (OS / 'wooden_spoon_trajectory.npz').unlink(); np.savez_compressed(OS / 'wooden_spoon_trajectory.npz', **tr)
else: changed = []
# per-frame deltas and ball evidence
anc = np.zeros(3)
import trimesh; anc = trimesh.load(SRC / 'assets/wooden_spoon.obj', force='mesh', process=False).centroid
dpos = np.linalg.norm(np.einsum('tij,j->ti', T[:, si, :3, :3] - T0[:, si, :3, :3], anc), axis=1) * 100
ev = []
for (f0, f1) in wins:
    g = [r for r in rows if f0 <= r['frame'] <= f1 and 'ours' in r]
    ev.append(dict(frames=[f0, f1], ball_frames=len(g), ball_detections=int(sum(r['n'] for r in g)), median_px=dict({k: float(np.median([r[k] for r in g])) if g else None for k in ('ours', 'ours_twin', 'new', 'ours_newroll')})))
changed_frames = np.flatnonzero(phi != 0)
rec = dict(episode=32, object='wooden_spoon', method='R_corr = R_ours * Rx(w*phi_new): NEW roll about our long body axis (our camera chain, our positions), smoothstep ramps centred on window edges',
           windows=per_window, ramp_s=ramp, frames_changed=int(len(changed_frames)), first_last_changed=[int(changed_frames.min()), int(changed_frames.max())] if len(changed_frames) else None,
           per_frame_roll_deg={int(k): round(float(phi[k]), 3) for k in changed_frames}, reference_shift=dict(max_rotation_deg=float(np.abs(phi).max()), mean_rotation_deg_over_episode=float(np.abs(phi).mean()),
           mean_rotation_deg_over_changed=float(np.abs(phi[changed_frames]).mean()) if len(changed_frames) else 0., max_anchor_shift_cm=float(dpos.max()), mean_anchor_shift_cm=float(dpos.mean())),
           ball_evidence=ev, ball_source=dict(script=str(S / 'code/roll_balls.py'), script_sha256=sha(S / 'code/roll_balls.py'), result=str(S / 'roll/roll_balls.json'), result_sha256=sha(S / 'roll/roll_balls.json'),
           detections='h2h_spoon/markers/balls_32_wooden_spoon.json', canonical='h2h_spoon/markers/canon.json', verdicts='h2h_spoon/verdicts.json (32: NEW f90-99, f115-129, f150-172; OURS f140-149 and outside f89-176)'),
           new_track='/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/h2h_spoon/inputs_new/episode_000032/wooden_spoon_trajectory.npz',
           new_track_sha256=sha('/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/h2h_spoon/inputs_new/episode_000032/wooden_spoon_trajectory.npz'),
           trajectory_npz_keys_rewritten=changed, original_object_poses_sha256=sha(SRC / 'object_poses.npz'), corrected_object_poses_sha256=sha(OS / 'object_poses.npz'),
           script=str(Path(__file__).resolve()), script_sha256=sha(__file__), cup_unchanged=True, positions_unchanged=True)
(OS / 'roll_correction.json').write_text(json.dumps(rec, indent=2))
# corrected episode dir
OE.mkdir(parents=True)
for p in EPD.iterdir(): shutil.copy2(p, OE / p.name)
m = json.loads((EPD / 'input_manifest.json').read_text()); Sm = np.asarray(m['setup_from_source'])
assert m['source'] == str(SRC) and m['source_object_poses_sha256'] == sha(SRC / 'object_poses.npz')
for fn in ('tracking_reference.npz', 'reference_trajectory.npz'):
    d = dict(np.load(EPD / fn, allow_pickle=False)); jn = [str(x) for x in d['joint_names']]
    a = jn.index('wooden_spoon_free') if 'wooden_spoon_free' in jn else None
    TS0, TS = Sm @ T0[:, si], Sm @ T[:, si]
    if fn == 'tracking_reference.npz':
        q = d['qpos']; qa = [i for i, x in enumerate(jn) if x.startswith('wooden_spoon')]
        adr = int(np.sum([7 if x.endswith('_free') else 1 for x in jn[:qa[0]]])) if False else None
    if 'object_positions' in d:
        oi = [str(x) for x in d['object_names']].index('wooden_spoon')
        q0 = R.from_matrix(TS0[:, :3, :3]).as_quat()[:, [3, 0, 1, 2]]; dq = np.minimum(np.abs(d['object_quaternions_wxyz'][:, oi] - q0).max(1), np.abs(d['object_quaternions_wxyz'][:, oi] + q0).max(1))
        assert np.abs(d['object_positions'][:, oi] - TS0[:, :3, 3]).max() < 1e-6 and dq.max() < 1e-6, ('object columns differ from setup_from_source @ poses', fn)
        qn = R.from_matrix(TS[:, :3, :3]).as_quat()[:, [3, 0, 1, 2]]; qn *= np.sign(np.sum(qn * d['object_quaternions_wxyz'][:, oi], 1))[:, None]
        d['object_quaternions_wxyz'][:, oi] = qn
    np.savez(OE / fn, **d) if False else None
    d_saved = d
    if fn == 'tracking_reference.npz': trk = d
    else: rtj = d
# tracking_reference qpos spoon columns: locate by matching the original pose
qpos = trk['qpos'].copy(); TS0, TS = Sm @ T0[:, si], Sm @ T[:, si]; adr = None
for c in range(qpos.shape[1] - 6):
    if np.abs(qpos[:, c:c + 3] - TS0[:, :3, 3]).max() < 1e-6:
        q0 = R.from_matrix(TS0[:, :3, :3]).as_quat()[:, [3, 0, 1, 2]]
        if np.minimum(np.abs(qpos[:, c + 3:c + 7] - q0).max(1), np.abs(qpos[:, c + 3:c + 7] + q0).max(1)).max() < 1e-6: adr = c; break
assert adr is not None, 'spoon columns not found in tracking_reference qpos'
qn = R.from_matrix(TS[:, :3, :3]).as_quat()[:, [3, 0, 1, 2]]; qn *= np.sign(np.sum(qn * qpos[:, adr + 3:adr + 7], 1))[:, None]; qpos[:, adr + 3:adr + 7] = qn
assert np.array_equal(qpos[0], trk['qpos'][0]), 'frame 0 must stay identical (scene initial state)'
trk['qpos'] = qpos; np.savez(OE / 'tracking_reference.npz', **trk)
if 'qpos' in rtj and rtj['qpos'].shape == trk['qpos'].shape and np.abs(rtj['qpos'][:, adr:adr + 3] - TS0[:, :3, 3]).max() < 1e-6:
    q = rtj['qpos'].copy(); q[:, adr + 3:adr + 7] = qn; rtj['qpos'] = q
np.savez(OE / 'reference_trajectory.npz', **rtj)
m.update(source=str(OS), source_reference=str(OS / 'reference.parquet'), source_reference_sha256=sha(OS / 'reference.parquet'), source_object_poses_sha256=sha(OS / 'object_poses.npz'))
m['spoon_roll_correction'] = dict(record=str(OS / 'roll_correction.json'), record_sha256=sha(OS / 'roll_correction.json'), script=str(Path(__file__).resolve()), script_sha256=sha(__file__),
    arguments=sys.argv[2:], original_manifest=str(EPD / 'input_manifest.json'), original_manifest_sha256=sha(EPD / 'input_manifest.json'), original_source=str(SRC),
    original_source_object_poses_sha256=sha(SRC / 'object_poses.npz'), tracking_reference_spoon_qpos_columns=[adr, adr + 7],
    note='spoon roll replaced by the NEW track roll in the ball-supported windows; positions, cup, frame 0, scene, meshes unchanged; tracking_reference/reference_trajectory spoon columns updated to the corrected source')
(OE / 'input_manifest.json').write_text(json.dumps(m, indent=2) + '\n')
print(json.dumps(dict(source=str(OS), episode_dir=str(OE), frames_changed=rec['frames_changed'], shift=rec['reference_shift'], windows=per_window, balls=ev), indent=1))
