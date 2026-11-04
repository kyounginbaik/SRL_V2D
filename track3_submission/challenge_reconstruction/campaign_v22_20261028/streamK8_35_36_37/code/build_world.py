"""Stream C7 (v22): re-target the wooden_spoon reference of ep35 / ep36 to target_final_v5 (our own reconstruction recovery).
  T_new(f) = G0 @ T_target(f) @ X
  X : scorer registration of our submitted (budget-decoded) mesh onto the target mesh (MP-SR._register_object_body, seed 1_000_003*ep,
      tlibC.registration; the target mesh is byte-identical to our mesh, X ~ identity up to the budget decimation).
  G0: one rigid transform per episode mapping the target frame-0 pose onto our physics anchor (the source frame-0 spoon pose = the
      keyframe start of the physics scene = achieved frame 0). Since the official scorer aligns every submission rigidly at frame 0
      (single object), the error of a rollout against T_new in our world equals its score vs target_final_v5 exactly.
  Gt (table-plane correction) = identity: any further rigid transform would break the frame-0 identity and with it the
      scorer-identity of the internal cost; end-rest height differences are recorded instead (rest_check).
Writes (new dirs only): streamK8_35_36_37/source/episode_X (copy of the episode's current source, object_poses.npz [+ wooden_spoon_trajectory.npz]
rewritten, target_retarget.json) and streamK8_35_36_37/episode/episode_X (copy of the actor's adapter episode dir with the spoon columns of
tracking_reference.npz [/ reference_trajectory.npz] replaced, input_manifest.json -> new source).
usage: build_world.py EP"""
import sys, json, shutil, hashlib, numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); import tlibK as L
S7 = HERE.parent; EP = int(sys.argv[1]); name = f'episode_{EP:06d}'
RD = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017')
EPD = {37: RD / 'episodes/episode_000037', 35: RD / 'refinement_20261018/wrist_chart/episode_000035/chart_0/adapter', 36: RD / 'refinement_20261021_ep36/episode_corrected/episode_000036'}[EP]
OUT, OEP = S7 / 'source' / name, S7 / 'episode' / name
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
if OUT.exists() or OEP.exists(): raise SystemExit('exists')
man = json.loads((EPD / 'input_manifest.json').read_text()); SRC = Path(man['source']); S = np.asarray(man['setup_from_source'])
assert sha(SRC / 'object_poses.npz') == man['source_object_poses_sha256']
z = dict(np.load(SRC / 'object_poses.npz', allow_pickle=False)); names = [str(x) for x in z['names']]; si = names.index('wooden_spoon')
T0 = z['world_from_object'].copy(); n = len(T0)
X, anchor, ca, rot = L.registration(EP); Tt = L.target(EP); assert len(Tt) == n, (len(Tt), n)
Tb = Tt @ X; G0 = T0[0, si] @ np.linalg.inv(Tb[0]); Tn = G0 @ Tb
pe, oe = L.errors(EP, Tn); assert pe.max() < 1e-4 and oe.max() < 1e-3, (pe.max(), oe.max())
shutil.copytree(SRC, OUT, symlinks=True); T = T0.copy(); T[:, si] = Tn; z['world_from_object'] = T; np.savez_compressed(OUT / 'object_poses.npz', **z)
changed = []
if (SRC / 'wooden_spoon_trajectory.npz').exists():
    tr = dict(np.load(SRC / 'wooden_spoon_trajectory.npz', allow_pickle=False))
    for key in list(tr):
        a = tr[key]
        if a.ndim == 3 and a.shape[1:] == (4, 4) and len(a) == n:
            tr[key] = np.einsum('nij,njk,nkl->nil', a, np.linalg.inv(T0[:, si]), Tn); changed.append(key)
    np.savez_compressed(OUT / 'wooden_spoon_trajectory.npz', **tr)
shutil.copytree(EPD, OEP, symlinks=True)
Ts = S @ Tn; qs = R.from_matrix(Ts[:, :3, :3]).as_quat(scalar_first=True); ps = Ts[:, :3, 3]
for k in range(1, n):
    if np.dot(qs[k], qs[k - 1]) < 0: qs[k] = -qs[k]
t = dict(np.load(EPD / 'tracking_reference.npz', allow_pickle=False)); jn = [str(x) for x in t['joint_names']]; c = jn.index('wooden_spoon_free[0]')
if np.dot(qs[0], t['qpos'][0, c + 3:c + 7]) < 0: qs = -qs
t['qpos'] = t['qpos'].copy(); t['qpos'][:, c:c + 3] = ps; t['qpos'][:, c + 3:c + 7] = qs; np.savez_compressed(OEP / 'tracking_reference.npz', **t)
if (EPD / 'reference_trajectory.npz').exists():
    rt = dict(np.load(EPD / 'reference_trajectory.npz', allow_pickle=False)); rt['qpos'] = rt['qpos'].copy(); rt['qpos'][:, c:c + 3] = ps; rt['qpos'][:, c + 3:c + 7] = qs
    op = rt['object_positions'].copy(); op[:, si] = ps; rt['object_positions'] = op
    oq = rt['object_quaternions_wxyz'].copy(); oq[:, si] = qs; rt['object_quaternions_wxyz'] = oq; np.savez_compressed(OEP / 'reference_trajectory.npz', **rt)
def shift(Ta, Tb):
    dp = np.linalg.norm(Ta[:, :3, 3] - Tb[:, :3, 3], axis=1) * 100; dr = np.degrees(R.from_matrix(np.einsum('nji,njk->nik', Ta[:, :3, :3], Tb[:, :3, :3])).magnitude()); return dp, dr
dpo, dro = shift(T0[:, si], Tn)
import trimesh
V = np.asarray(trimesh.load(L.TGT + f'/{name}/wooden_spoon_mesh_metric.obj', force='mesh').vertices)
low = lambda T: float(((S @ T)[:3, :3] @ V.T + (S @ T)[:3, 3:4])[2].min() * 100)
TGTF = Path(L.TGT) / name / 'wooden_spoon_trajectory.npz'
rec = dict(episode=EP, object='wooden_spoon', schema='streamK8_target_final_v5_retarget_v1',
    method='T_new(f) = G0 @ T_target_final_v5(f) @ X; X = official scorer registration of our submitted mesh onto the target mesh; G0 = frame-0 anchor onto our physics start (= scorer frame-0 alignment); Gt = identity; all frames replaced, no clipping',
    target_track=str(TGTF), target_track_sha256=sha(TGTF), target_mesh_sha256=sha(Path(L.TGT) / name / 'wooden_spoon_mesh_metric.obj'),
    target_bundle='/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/selected20_final_v5/' + name,
    X=X.tolist(), X_rotation_deg=float(np.degrees(R.from_matrix(X[:3, :3]).magnitude())), X_translation_cm=(X[:3, 3] * 100).tolist(), scorer_anchor_in_our_body_m=ca.tolist(),
    G0_source_from_targetworld=G0.tolist(), self_check_scorer_identical=dict(max_pos_cm=float(pe.max()), max_ori_deg=float(oe.max())),
    rest_check_lowest_mesh_point_setup_cm=dict(frames=[0, 10, n - 30, n - 10, n - 1], old=[round(low(T0[k, si]), 2) for k in (0, 10, n - 30, n - 10, n - 1)], new=[round(low(Tn[k]), 2) for k in (0, 10, n - 30, n - 10, n - 1)]),
    shift_vs_previous_reference=dict(source=str(SRC / 'object_poses.npz'), sha256=sha(SRC / 'object_poses.npz'), mean_cm=float(dpo.mean()), max_cm=float(dpo.max()), mean_deg=float(dro.mean()), max_deg=float(dro.max()),
        per_frame_cm=np.round(dpo, 3).tolist(), per_frame_deg=np.round(dro, 2).tolist()),
    trajectory_keys_rewritten=changed, original_object_poses_sha256=sha(SRC / 'object_poses.npz'), new_object_poses_sha256=sha(OUT / 'object_poses.npz'),
    original_episode_dir=str(EPD), script=str(Path(__file__).resolve()), script_sha256=sha(__file__), tlib_sha256=sha(HERE / 'tlibK.py'))
(OUT / 'target_retarget.json').write_text(json.dumps(rec, indent=1))
man['source'] = str(OUT); man['source_reference'] = str(OUT / 'reference.parquet') if (OUT / 'reference.parquet').exists() else man.get('source_reference')
if (OUT / 'reference.parquet').exists(): man['source_reference_sha256'] = sha(OUT / 'reference.parquet')
man['source_object_poses_sha256'] = sha(OUT / 'object_poses.npz')
man['corrected_track_note'] = 'v22: wooden_spoon reference = target_final_v5 (our reconstruction recovery) via G0 @ T @ X, see source/' + name + '/target_retarget.json; robot/scene unchanged'
man['target_retarget'] = dict(record=str(OUT / 'target_retarget.json'), record_sha256=sha(OUT / 'target_retarget.json'), original_source=str(SRC), original_source_object_poses_sha256=sha(SRC / 'object_poses.npz'), original_manifest=str(EPD / 'input_manifest.json'), original_manifest_sha256=sha(EPD / 'input_manifest.json'))
(OEP / 'input_manifest.json').write_text(json.dumps(man, indent=2) + '\n')
print(json.dumps({k: rec[k] for k in ('X_rotation_deg', 'X_translation_cm', 'self_check_scorer_identical', 'rest_check_lowest_mesh_point_setup_cm', 'trajectory_keys_rewritten')}, indent=1))
print('shift vs previous', dpo.mean(), dro.mean())
