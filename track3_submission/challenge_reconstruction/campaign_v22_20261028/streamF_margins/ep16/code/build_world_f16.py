"""Stream F (v22) ep16: re-target BOTH object references of world_6prong_newtrack (streamN phase 5 world) to target_final_v5.
  T_new[o](f) = G0 @ T_target[o](f) @ X[o]   (f >= 1)
  X[o] : the scorer's registration of our shipped (budget-encoded) mesh onto the target mesh (tlib16.registration; seed 1_000_003*16+101*slot)
  G0   : maps the target RACK (slot 0 = scorer anchor) frame-0 pose onto our physics rack frame-0 pose (= keyframe = achieved frame 0).
         The scorer aligns the whole scene rigidly with the rack frame-0 pose, so errors against T_new equal score_any errors
         (frame 0 excepted, see below). Gt = identity (would break that identity).
  Frame 0 of both objects is kept at the physics start pose (the CEM requires scene keyframe == reference frame 0; the scene is
  unchanged: same physics, same scene sha256). Rack frame 0 differs from G0 T X by ~1e-15; spoon frame 0 by the recorded rest offset.
Writes (new dirs only): ep16/world_T/source/episode_000016 (copy of the streamN source with object_poses.npz, *_trajectory.npz,
reference.parquet rewritten + target_retarget.json + spoon_track_retarget.json (the stage wrapper's correction record format)),
ep16/world_T/episodes/episode_000016 (copy of the episode dir: tracking_reference / reference_trajectory object columns, manifest).
usage: build_world_f16.py"""
import sys, json, shutil, hashlib, numpy as np, pyarrow.parquet as pq, pyarrow as pa
from pathlib import Path
from scipy.spatial.transform import Rotation as R
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); import tlib16 as L
W0 = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamN_ep16/world_6prong_newtrack')
WT = HERE.parent / 'world_T'; name = 'episode_000016'
EPD = W0 / 'episodes' / name; man = json.loads((EPD / 'input_manifest.json').read_text()); SRC = Path(man['source']).resolve()
OS, OE = WT / 'source' / name, WT / 'episodes' / name
if OS.exists() or OE.exists(): raise SystemExit('exists')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(SRC / 'object_poses.npz') == man['source_object_poses_sha256']
S = np.asarray(man['setup_from_source']); assert np.allclose(S, np.eye(4))
z = dict(np.load(SRC / 'object_poses.npz', allow_pickle=False)); names = [str(x) for x in z['names']]; assert names == L.NAMES, names
T0 = z['world_from_object'].copy(); n = len(T0)
Tb = L.target_ourbody(); assert all(len(Tb[o]) == n for o in names)
G0 = T0[0, 0] @ np.linalg.inv(Tb['plastic_dish_rack'][0])
new = {o: G0 @ Tb[o] for o in names}
d_rack0 = np.abs(new['plastic_dish_rack'][0] - T0[0, 0]).max(); assert d_rack0 < 1e-9, d_rack0
sp0_cm = float(np.linalg.norm(new['wooden_spoon'][0, :3, 3] - T0[0, 1, :3, 3]) * 100)
sp0_deg = float(np.degrees(R.from_matrix(new['wooden_spoon'][0, :3, :3].T @ T0[0, 1, :3, :3]).magnitude()))
for i, o in enumerate(names): new[o][0] = T0[0, i]
e = L.errors(new); selfcheck = {o: dict(max_cm_f1plus=float(p[1:].max()), max_deg_f1plus=float(d[1:].max()), frame0_cm=float(p[0]), frame0_deg=float(d[0])) for o, (p, d) in e.items()}
assert all(v['max_cm_f1plus'] < 1e-4 and v['max_deg_f1plus'] < 1e-3 for v in selfcheck.values()), selfcheck
Wn = np.stack([new[o] for o in names], 1)
# --- source copy
shutil.copytree(SRC, OS, symlinks=True)
z['world_from_object'] = Wn; (OS / 'object_poses.npz').unlink(); np.savez_compressed(OS / 'object_poses.npz', **z)
for i, o in enumerate(names):
    tr = dict(np.load(SRC / f'{o}_trajectory.npz', allow_pickle=False)); tr['T_world_obj'] = Wn[:, i].copy(); tr['T_rgb_obj'] = np.linalg.inv(z['world_from_rgb']) @ Wn[:, i]
    (OS / f'{o}_trajectory.npz').unlink(); np.savez_compressed(OS / f'{o}_trajectory.npz', **tr)
tab = pq.read_table(SRC / 'reference.parquet'); rows = tab.to_pylist(); assert len(rows) == 1
old_pos = np.asarray(rows[0]['object_body_position']); assert old_pos.shape == (n, 2, 3) and np.allclose(old_pos, T0[:, :, :3, 3], atol=1e-6), old_pos.shape
rows[0]['object_body_position'] = Wn[:, :, :3, 3].tolist()
rows[0]['object_body_wxyz'] = R.from_matrix(Wn[:, :, :3, :3].reshape(-1, 3, 3)).as_quat(scalar_first=True).reshape(n, 2, 4).tolist()
(OS / 'reference.parquet').unlink(); pq.write_table(pa.Table.from_pylist(rows, schema=tab.schema), OS / 'reference.parquet')
# --- episode dir copy
shutil.copytree(EPD, OE, symlinks=True)
qn = {}
for o in names:
    q = R.from_matrix(new[o][:, :3, :3]).as_quat(scalar_first=True)
    for k in range(1, n):
        if np.dot(q[k], q[k - 1]) < 0: q[k] = -q[k]
    qn[o] = q
def fix(Q, jn):
    Q = Q.copy()
    for o in names:
        c = jn.index(f'{o}_free[0]'); old = Q[:, c:c + 7].copy()
        assert np.allclose(old[:, :3], T0[:, names.index(o), :3, 3], atol=1e-6)
        q = qn[o] * (1 if np.dot(qn[o][0], old[0, 3:]) >= 0 else -1)
        Q[:, c:c + 3] = new[o][:, :3, 3]; Q[:, c + 3:c + 7] = q; Q[0, c:c + 7] = old[0]
    return Q
t = dict(np.load(EPD / 'tracking_reference.npz', allow_pickle=False)); jn = [str(x) for x in t['joint_names']]
t['qpos'] = fix(t['qpos'], jn); (OE / 'tracking_reference.npz').unlink(); np.savez_compressed(OE / 'tracking_reference.npz', **t)
rt = dict(np.load(EPD / 'reference_trajectory.npz', allow_pickle=False)); assert [str(x) for x in rt['joint_names']] == jn
rt['qpos'] = fix(rt['qpos'], jn)
for i, o in enumerate(names):
    c = jn.index(f'{o}_free[0]'); rt['object_positions'][:, i] = rt['qpos'][:, c:c + 3]; rt['object_quaternions_wxyz'][:, i] = rt['qpos'][:, c + 3:c + 7]
(OE / 'reference_trajectory.npz').unlink(); np.savez_compressed(OE / 'reference_trajectory.npz', **rt)
def shift(Ta, Tb_):
    dp = np.linalg.norm(Ta[:, :3, 3] - Tb_[:, :3, 3], axis=1) * 100; dr = np.degrees(R.from_matrix(np.einsum('nji,njk->nik', Ta[:, :3, :3], Tb_[:, :3, :3])).magnitude()); return dp, dr
sh = {}
for i, o in enumerate(names):
    dp, dr = shift(T0[:, i], new[o]); sh[o] = dict(mean_cm=float(dp.mean()), max_cm=float(dp.max()), mean_deg=float(dr.mean()), max_deg=float(dr.max()),
        per_second_cm=[round(float(dp[s:s + 20].mean()), 2) for s in range(0, n, 20)], per_second_deg=[round(float(dr[s:s + 20].mean()), 1) for s in range(0, n, 20)])
ZIP = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/source/final_recon_ours/episode_000016/object_poses.npz')
reg = L.registration()
rec = dict(episode=16, schema='streamF_v22_target_final_v5_retarget_v1', objects=names,
    method='T_new[o](f) = G0 @ T_target_final_v5[o](f) @ X[o] for f >= 1; frame 0 = physics start (scene unchanged); X = scorer registration of the shipped mesh onto the target mesh; G0 = rack (scorer anchor slot 0) frame-0 alignment; Gt = identity',
    target_dir=L.TGT, target_sha256={o: sha(f'{L.TGT}/{o}_trajectory.npz') for o in names} | {o + '_mesh': sha(f'{L.TGT}/{o}_mesh_metric.obj') for o in names},
    target_bundle='/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/selected20_final_v5/episode_000016',
    X={o: reg[o][0].tolist() for o in names}, X_rotation_deg={o: float(np.degrees(R.from_matrix(reg[o][0][:3, :3]).magnitude())) for o in names},
    scorer_anchor_in_our_body_m={o: reg[o][2].tolist() for o in names}, G0_setup_from_targetworld=G0.tolist(),
    frame0_kept=dict(rack_max_abs_diff=float(d_rack0), spoon_target_vs_physics_start_cm=sp0_cm, spoon_target_vs_physics_start_deg=sp0_deg),
    self_check_scorer_identical=selfcheck, shift_vs_previous_reference=sh,
    previous_source=str(SRC), previous_object_poses_sha256=sha(SRC / 'object_poses.npz'),
    original_object_poses_sha256=sha(ZIP), user_zip_object_poses=str(ZIP), corrected_object_poses_sha256=sha(OS / 'object_poses.npz'),
    original_episode_dir=str(EPD), script=str(Path(__file__).resolve()), script_sha256=sha(__file__), tlib_sha256=sha(HERE / 'tlib16.py'))
(OS / 'target_retarget.json').write_text(json.dumps(rec, indent=1))
(OS / 'spoon_track_retarget.json').unlink(missing_ok=True); (OS / 'spoon_track_retarget.json').write_text(json.dumps(rec, indent=1))
man['source'] = str(OS); man['source_reference'] = str(OS / 'reference.parquet'); man['source_reference_sha256'] = sha(OS / 'reference.parquet')
man['source_object_poses_sha256'] = sha(OS / 'object_poses.npz')
man['new_input_provenance'] = dict(man['new_input_provenance'], hawor_source=str(OS), previous_hawor_source=man['new_input_provenance']['hawor_source'])
man['corrected_track_note'] = 'v22 stream F: rack + spoon references = target_final_v5 (our reconstruction recovery) via G0 @ T @ X, rack frame-0 anchor; scene/robot/meshes unchanged; see source/episode_000016/target_retarget.json'
(OE / 'input_manifest.json').write_text(json.dumps(man, indent=2) + '\n')
print(json.dumps(dict(frame0=rec['frame0_kept'], selfcheck=selfcheck, shift={o: (round(v['mean_cm'], 2), round(v['mean_deg'], 1), round(v['max_cm'], 1)) for o, v in sh.items()}, X_deg=rec['X_rotation_deg']), indent=1))
