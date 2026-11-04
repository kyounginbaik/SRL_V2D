#!/usr/bin/env python3
"""Stream N phase 5: re-target the episode-16 reference by replacing the wooden_spoon track with the coworker's NEW track
(h2h_spoon/inputs_new/episode_000016/wooden_spoon_trajectory.npz, verdict NEW/high in h2h_spoon/verdicts.json), expressed in our
world frame through the shared camera chain. Rack track, hands, camera, clock untouched (byte-identical columns).

Frames: NEW world = rectified-left camera at frame 0 (T_world_obj), camera cam0_T_cam (rectL), T_a_from_rectL (ego_cam_a <- rectL).
Ours (hawor_stereo_recon_v2/episode_000016): world_from_rgb, rgb = ego_cam_a undistorted (same K as K_a_undist).
G = world_from_rgb[k] @ inv(cam0_T_cam[k] @ inv(T_a_from_rectL)) must be constant over k (checked; spread reported) and
T_ours_world_spoon = G @ T_new_world_spoon @ M, where M maps OUR spoon body frame to the NEW mesh body frame (p_new = M p_ours).
M is taken as identity when the registration of our mesh 6cc4d2b2.. onto their wooden_spoon_mesh_metric.obj is identity within
tolerance (checked: rotation, translation, Chamfer at identity vs best); otherwise the fixed registration is applied.
usage: retarget_spoon_track.py OUT_SOURCE_ROOT   (writes OUT_SOURCE_ROOT/episode_000016 + spoon_track_retarget.json)"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import sys, json, shutil, hashlib
from pathlib import Path
import numpy as np, trimesh, pyarrow as pa, pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/h2h_spoon/code')
import meshreg
NEW = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/h2h_spoon/inputs_new/episode_000016')
SRC = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/hawor_stereo_recon_v2/episode_000016')
OBJ = 'wooden_spoon'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
out_root = Path(sys.argv[1]); dst = out_root / 'episode_000016'
if dst.exists(): raise SystemExit('exists, never overwrite: ' + str(dst))
shutil.copytree(SRC, dst, ignore=shutil.ignore_patterns('archive_wilor', 'archive_before_stereo', '*.mp4', '__pycache__'))
# 1. mesh body frames
mo = trimesh.load(SRC / f'assets/{OBJ}_visual.obj', force='mesh'); mn = trimesh.load(NEW / f'{OBJ}_mesh_metric.obj', force='mesh')
reg = meshreg.register(mo, mn); Mbest, cbest = reg[0]
A = meshreg._samp(mo); B = meshreg._samp(mn, seed=1); cid = (cKDTree(B).query(A)[0].mean() + cKDTree(A).query(B)[0].mean()) / 2
rot_best = float(np.degrees(R.from_matrix(Mbest[:3, :3]).magnitude())); t_best = float(np.linalg.norm(Mbest[:3, 3]) * 100)
use_identity = rot_best < 0.5 and t_best < 0.1 and cid - cbest < 2e-4
M = np.eye(4) if use_identity else Mbest
# 2. world transform via the shared camera chain
cn = np.load(NEW / 'camera_motion.npz'); co = np.load(SRC / 'camera_motion.npz')
Gk = np.array([co['world_from_rgb'][k] @ np.linalg.inv(cn['cam0_T_cam'][k] @ np.linalg.inv(cn['T_a_from_rectL'])) for k in range(len(co['world_from_rgb']))])
G = Gk[0]; g_dp = np.linalg.norm(Gk[:, :3, 3] - G[:3, 3], axis=1) * 100; g_dr = np.degrees(R.from_matrix(Gk[:, :3, :3] @ G[:3, :3].T).magnitude())
if g_dp.max() > 0.01 or g_dr.max() > 0.01: raise SystemExit(f'camera chains are not related by one constant transform: {g_dp.max()} cm {g_dr.max()} deg')
# 3. new spoon poses in our world
z = dict(np.load(SRC / 'object_poses.npz')); names = [str(x) for x in z['names']]; i = names.index(OBJ); T = z['world_from_object'].copy()
Tn = np.load(NEW / f'{OBJ}_trajectory.npz')['T_world_obj']; assert len(Tn) == len(T)
Tnew = np.einsum('ij,kjl,lm->kim', G, Tn, M)
diff_cm = np.linalg.norm(Tnew[:, :3, 3] - T[:, i, :3, 3], axis=1) * 100; diff_deg = np.degrees(R.from_matrix(Tnew[:, :3, :3] @ np.transpose(T[:, i, :3, :3], (0, 2, 1))).magnitude())
along = np.einsum('kj,kj->k', Tnew[:, :3, 3] - T[:, i, :3, 3], T[:, i, :3, 0]) * 100
rack = names.index('plastic_dish_rack'); Tr_new = G @ np.load(NEW / 'plastic_dish_rack_trajectory.npz')['T_world_obj'][0]
rack_cm = float(np.linalg.norm(Tr_new[:3, 3] - T[0, rack, :3, 3]) * 100); rack_deg = float(np.degrees(R.from_matrix(Tr_new[:3, :3] @ T[0, rack, :3, :3].T).magnitude()))
T[:, i] = Tnew; z['world_from_object'] = T; np.savez_compressed(dst / 'object_poses.npz', **z)
tr = dict(np.load(SRC / f'{OBJ}_trajectory.npz')); tr['T_world_obj'] = Tnew; tr['T_rgb_obj'] = np.linalg.inv(z['world_from_rgb']) @ Tnew
np.savez_compressed(dst / f'{OBJ}_trajectory.npz', **tr)
tab = pq.read_table(SRC / 'reference.parquet'); row = tab.to_pylist()[0]; bn = list(row['object_body_names']); j = bn.index(OBJ)
pos = np.array(row['object_body_position']); q = np.array(row['object_body_wxyz'])
pos[:, j] = Tnew[:, :3, 3]; qq = R.from_matrix(Tnew[:, :3, :3]).as_quat(); q[:, j] = qq[:, [3, 0, 1, 2]]
if bn[0] == OBJ:
    row['object_root_position'] = pos[:, 0].tolist(); row['object_root_axis_angle'] = R.from_matrix(Tnew[:, :3, :3]).as_rotvec().tolist()
row['object_body_position'] = pos.tolist(); row['object_body_wxyz'] = q.tolist()
pq.write_table(pa.Table.from_pylist([row], schema=tab.schema), dst / 'reference.parquet')
zz = np.load(dst / 'object_poses.npz'); rack_unchanged = bool(np.array_equal(zz['world_from_object'][:, rack], np.load(SRC / 'object_poses.npz')['world_from_object'][:, rack]))
rec = dict(schema='spoon_track_retarget_v1', episode=16, object=OBJ,
    reason='h2h_spoon phase-5 verdict NEW/high for ep16 wooden_spoon: canonical mocap balls 44.2 px (ours) vs 9.0 px (new), inl10 0.01 vs 0.55, IoU 0.62 vs 0.75, size depth scale 1.100 vs 1.015; rack TIE (kept ours)',
    new_track=dict(path=str(NEW / f'{OBJ}_trajectory.npz'), sha256=sha(NEW / f'{OBJ}_trajectory.npz'), camera=str(NEW / 'camera_motion.npz'), camera_sha256=sha(NEW / 'camera_motion.npz'), readme=(NEW / 'README.md').read_text()),
    world_transform_G_new_to_ours=G, camera_chain_constancy=dict(max_dpos_cm=float(g_dp.max()), max_drot_deg=float(g_dr.max()), formula='world_from_rgb[k] @ inv(cam0_T_cam[k] @ inv(T_a_from_rectL))'),
    mesh_registration=dict(ours=str(SRC / f'assets/{OBJ}_visual.obj'), ours_sha256=sha(SRC / f'assets/{OBJ}_visual.obj'), new=str(NEW / f'{OBJ}_mesh_metric.obj'), new_sha256=sha(NEW / f'{OBJ}_mesh_metric.obj'),
        best_rotation_deg=rot_best, best_translation_cm=t_best, best_chamfer_mm=float(cbest * 1000), identity_chamfer_mm=float(cid * 1000), second_best_chamfer_mm=float(reg[1][1] * 1000),
        applied='identity (body frames match)' if use_identity else 'best registration', M_applied=M, their_mesh_shipped=False),
    residual_new_vs_old_spoon=dict(mean_cm=float(diff_cm.mean()), mean_deg=float(diff_deg.mean()), min_cm=float(diff_cm.min()), at_rest_frames_0_14_cm=float(diff_cm[:15].mean()), at_rest_frames_0_14_deg=float(diff_deg[:15].mean()),
        along_handle_cm_mean=float(along.mean()), per_second={f'{k/20:.1f}': [round(float(diff_cm[k]), 2), round(float(diff_deg[k]), 1), round(float(along[k]), 2)] for k in range(0, len(T), 10)}),
    rack_check=dict(note='rack kept as ours (static); the NEW rack track (other mesh) mapped by G sits this far from ours', offset_cm=rack_cm, offset_deg=rack_deg),
    original_object_poses_sha256=sha(SRC / 'object_poses.npz'), corrected_object_poses_sha256=sha(dst / 'object_poses.npz'),
    original_reference_parquet_sha256=sha(SRC / 'reference.parquet'), corrected_reference_parquet_sha256=sha(dst / 'reference.parquet'),
    rack_columns_unchanged=rack_unchanged, script=str(Path(__file__).resolve()), script_sha256=sha(__file__))
(dst / 'spoon_track_retarget.json').write_text(json.dumps(rec, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)) + '\n')
print(json.dumps({k: rec[k] for k in ('camera_chain_constancy', 'residual_new_vs_old_spoon', 'rack_check', 'rack_columns_unchanged')}, default=str)[:1500])
print('mesh', {k: rec['mesh_registration'][k] for k in ('best_rotation_deg', 'best_translation_cm', 'best_chamfer_mm', 'identity_chamfer_mm', 'applied')})
