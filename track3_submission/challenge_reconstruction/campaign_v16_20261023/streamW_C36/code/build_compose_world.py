"""Stream W (phase 6, v21C), episode 36: reference = the coworker's compose_eval spoon track, expressed for OUR mesh body frame
(c9ab88e33e5c) in OUR physics/source world.

  T_new(f) = A0 @ T_compose(f) @ Gc
  Gc: fixed mesh-to-mesh registration, exactly the official scorer's _register_object_body of our mesh onto theirs (their
      surface-anchor frame == our body pose @ [brot^T, ca]); 1.1 deg / 0.6 cm (slide along the handle), see record.
  A0: one rigid transform for the episode = frame-0 anchor (A0 = T_ours_src(0) @ inv(T_compose(0) @ Gc)). This is exactly the
      scorer's anchor frame-0 alignment, so a rollout scored against T_new in our world equals its score vs the compose track.
      Frame 0 is therefore byte-identical in pose to our source -> scene/keyframe unchanged. Camera-chain cross-check in record.
Positions AND orientations of every frame are replaced (no clipping in the reference; robot-side clipping is a plan choice).
Writes (new dirs only): source_compose/episode_000036 (copy of final_recon/episode_000036, object_poses.npz +
wooden_spoon_trajectory.npz rewritten, compose_retarget.json) and episode_compose/episode_000036 (copy of the legacy episode dir
with tracking_reference.npz / reference_trajectory.npz spoon pose replaced, input_manifest.json -> source_compose)."""
import sys, json, shutil, hashlib, numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
W = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamW_C36'); sys.path.insert(0, str(W / 'code')); import perframe as P
SRC = Path('/data1/SRL_V2D/challenge_reconstruction/final_recon/episode_000036')
EPD = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/episodes/episode_000036')
CORR = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep36/source_corrected/episode_000036/object_poses.npz')
OUT, OEP = W / 'source_compose/episode_000036', W / 'episode_compose/episode_000036'
CT = Path(P.CMP)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
if OUT.exists() or OEP.exists(): raise SystemExit('exists')
z = dict(np.load(SRC / 'object_poses.npz', allow_pickle=False)); names = [str(x) for x in z['names']]; si = names.index('wooden_spoon')
T0 = z['world_from_object'].copy(); n = len(T0); assert len(P.TC) == n
Gc = P.Gc(); Tc = P.TC @ Gc; A0 = T0[0, si] @ np.linalg.inv(Tc[0]); Tn = A0 @ Tc
T = T0.copy(); T[:, si] = Tn
pe, oe = P.errors(Tn); assert pe.max() < 1e-4 and oe.max() < 1e-3, (pe.max(), oe.max())
shutil.copytree(SRC, OUT, symlinks=True); z['world_from_object'] = T; np.savez_compressed(OUT / 'object_poses.npz', **z)
changed = []
if (OUT / 'wooden_spoon_trajectory.npz').exists():
    tr = dict(np.load(SRC / 'wooden_spoon_trajectory.npz', allow_pickle=False))
    for key in list(tr):
        a = tr[key]
        if a.ndim == 3 and a.shape[1:] == (4, 4) and len(a) == n:
            # same per-frame left-multiplied change as the world poses: new = a @ inv(T0) @ Tn  (keeps each key's own frame)
            tr[key] = np.einsum('nij,njk,nkl->nil', a, np.linalg.inv(T0[:, si]), Tn); changed.append(key)
    np.savez_compressed(OUT / 'wooden_spoon_trajectory.npz', **tr)
shutil.copytree(EPD, OEP, symlinks=True); man = json.loads((EPD / 'input_manifest.json').read_text()); S = np.asarray(man['setup_from_source'])
Ts = S @ Tn; qs = R.from_matrix(Ts[:, :3, :3]).as_quat(scalar_first=True); ps = Ts[:, :3, 3]
t = dict(np.load(EPD / 'tracking_reference.npz', allow_pickle=False)); jn = [str(x) for x in t['joint_names']]; c = jn.index('wooden_spoon_free[0]')
old = t['qpos'][:, c + 3:c + 7]; qs = qs * np.sign((qs * old).sum(1, keepdims=True))
for k in range(1, n): qs[k] *= np.sign((qs[k] * qs[k - 1]).sum()) if k and False else 1
t['qpos'] = t['qpos'].copy(); t['qpos'][:, c:c + 3] = ps; t['qpos'][:, c + 3:c + 7] = qs; np.savez_compressed(OEP / 'tracking_reference.npz', **t)
rt = dict(np.load(EPD / 'reference_trajectory.npz', allow_pickle=False)); rt['qpos'] = rt['qpos'].copy(); rt['qpos'][:, c:c + 3] = ps; rt['qpos'][:, c + 3:c + 7] = qs
op = rt['object_positions'].copy(); op[:, si] = ps; rt['object_positions'] = op
oq = rt['object_quaternions_wxyz'].copy(); oq[:, si] = qs; rt['object_quaternions_wxyz'] = oq; np.savez_compressed(OEP / 'reference_trajectory.npz', **rt)
# lineage numbers
cm = np.load(CT / 'camera_motion.npz'); Wr = z['world_from_rgb']; X = np.array([np.linalg.inv(A0 @ cm['cam0_T_cam'][f]) @ Wr[f] for f in range(n)])
xr = [float(np.degrees(R.from_matrix(X[f, :3, :3] @ X[0, :3, :3].T).magnitude())) for f in range(n)]; xt = [float(np.linalg.norm(X[f, :3, 3] - X[0, :3, 3]) * 100) for f in range(n)]
corr = np.load(CORR)['world_from_object'][:, 0]
def shift(Ta, Tb):
    dp = np.linalg.norm(Ta[:, :3, 3] - Tb[:, :3, 3], axis=1) * 100; dr = np.degrees(R.from_matrix(np.einsum('nji,njk->nik', Ta[:, :3, :3], Tb[:, :3, :3])).magnitude()); return dp, dr
dpo, dro = shift(T0[:, si], Tn); dpc, drc = shift(corr, Tn)
rec = dict(episode=36, object='wooden_spoon', schema='streamW_compose_retarget_v1',
    method='T_new(f) = A0 @ T_compose(f) @ Gc; Gc = official scorer registration of our mesh onto the compose mesh; A0 = frame-0 anchor (identical to the scorer alignment); all 407 frames replaced, no clipping',
    compose_track=str(CT / 'wooden_spoon_trajectory.npz'), compose_track_sha256=sha(CT / 'wooden_spoon_trajectory.npz'), compose_mesh=str(CT / 'wooden_spoon_mesh_metric.obj'), compose_mesh_sha256=sha(CT / 'wooden_spoon_mesh_metric.obj'),
    our_mesh=P.OURMESH, our_mesh_sha256=sha(P.OURMESH), Gc=Gc.tolist(), Gc_rotation_deg=float(np.degrees(R.from_matrix(Gc[:3, :3]).magnitude())), Gc_translation_cm=(Gc[:3, 3] * 100).tolist(),
    A0_source_from_composeworld=A0.tolist(),
    camera_chain_check=dict(note='X(f) = inv(A0 @ cam0_T_cam(f)) @ world_from_rgb(f) would be constant if both camera paths agree after the frame-0 anchor; constant part (11 deg) = camera-model/rectification + frame-0 object pose disagreement',
        rot_drift_deg_max=max(xr), rot_drift_deg_median=float(np.median(xr)), trans_drift_cm_max=max(xt), trans_drift_cm_median=float(np.median(xt)), X0_rotation_deg=float(np.degrees(R.from_matrix(X[0, :3, :3]).magnitude()))),
    self_check_vs_compose_scorer_identical=dict(max_pos_cm=float(pe.max()), max_ori_deg=float(oe.max())),
    shift_vs_original_source=dict(mean_cm=float(dpo.mean()), max_cm=float(dpo.max()), mean_deg=float(dro.mean()), max_deg=float(dro.max()), per_frame_cm=np.round(dpo, 3).tolist(), per_frame_deg=np.round(dro, 2).tolist()),
    shift_vs_v21_reference_corrected=dict(source=str(CORR), sha256=sha(CORR), mean_cm=float(dpc.mean()), max_cm=float(dpc.max()), mean_deg=float(drc.mean()), max_deg=float(drc.max()), per_frame_cm=np.round(dpc, 3).tolist(), per_frame_deg=np.round(drc, 2).tolist()),
    table_note='frame 0 identical to our source (spoon resting on our table); the compose final rest (f380-406) sits 1.0-1.4 cm higher and is rolled ~18 deg / pitched ~9 deg vs our final rest after the frame-0 anchor -> not a physical resting pose on our table; table NOT rebuilt (would move the frame-0 rest)',
    trajectory_keys_rewritten=changed, original_object_poses_sha256=sha(SRC / 'object_poses.npz'), new_object_poses_sha256=sha(OUT / 'object_poses.npz'),
    script=str(Path(__file__).resolve()), script_sha256=sha(__file__), perframe_sha256=sha(W / 'code/perframe.py'))
(OUT / 'compose_retarget.json').write_text(json.dumps(rec, indent=1))
man['source'] = str(OUT); man['source_reference'] = str(OUT / 'reference.parquet'); man['source_reference_sha256'] = sha(OUT / 'reference.parquet'); man['source_object_poses_sha256'] = sha(OUT / 'object_poses.npz')
man['corrected_track_note'] = 'v21C hedge: wooden_spoon reference = coworker compose_eval track (A0 @ T_compose @ Gc), see source_compose/episode_000036/compose_retarget.json; robot/scene unchanged'
man['compose_retarget'] = dict(record=str(OUT / 'compose_retarget.json'), record_sha256=sha(OUT / 'compose_retarget.json'), original_source=str(SRC), original_source_object_poses_sha256=sha(SRC / 'object_poses.npz'), original_manifest=str(EPD / 'input_manifest.json'), original_manifest_sha256=sha(EPD / 'input_manifest.json'))
(OEP / 'input_manifest.json').write_text(json.dumps(man, indent=2) + '\n')
print(json.dumps({k: rec[k] for k in ('Gc_rotation_deg', 'Gc_translation_cm', 'camera_chain_check', 'self_check_vs_compose_scorer_identical', 'new_object_poses_sha256', 'trajectory_keys_rewritten')}, indent=1))
print('shift vs orig', dpo.mean(), dro.mean(), 'vs corr', dpc.mean(), drc.mean())
