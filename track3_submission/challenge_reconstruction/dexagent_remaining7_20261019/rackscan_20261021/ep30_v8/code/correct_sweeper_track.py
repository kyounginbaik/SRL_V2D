"""Episode 30: undo the sweeper track's half-turn twin flip (2.5-3.2 s). The flat brush is symmetric under a half turn about
its long axis; the tracker jumped to the twin while the human wrist turned only ~33 deg. Corrected: frames <= 2.5 s unchanged,
frames >= 3.2 s = twin of the tracked pose (half turn about the body long axis through the mesh centre; agrees with a rigid hand
hold within 18-30 deg), geodesic blend in between. The mesh-centre trajectory is unchanged. Rack and all other files unchanged."""
import shutil, json, hashlib, sys
from pathlib import Path
import numpy as np, trimesh, pyarrow as pa, pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R, Slerp
W = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021'); W30 = W / 'ep30_v8'
src = W / 'collision_recon/episode_000030'; dst = W30 / 'collision_recon/episode_000030'
if dst.exists(): raise SystemExit('exists')
shutil.copytree(src, dst)
z = dict(np.load(src / 'object_poses.npz')); T = z['world_from_object'].copy(); names = [str(x) for x in z['names']]; i = names.index('mini_sweeper')
V = np.asarray(trimesh.load(src / 'assets/mini_sweeper_visual.obj', force='mesh', process=False).vertices); c = V.mean(0); w, U = np.linalg.eigh((V - c).T @ (V - c)); L = U[:, 2]
F = R.from_rotvec(np.pi * L).as_matrix(); t = np.arange(len(T)) / 20.; t0, t1 = 2.5, 3.2
cw = T[:, i, :3, :3] @ c + T[:, i, :3, 3]                      # mesh-centre trajectory (kept)
Rn = T[:, i, :3, :3].copy(); k0, k1 = int(t0 * 20), int(t1 * 20)
Rn[k1:] = T[k1:, i, :3, :3] @ F
sl = Slerp([t0, t1], R.from_matrix([T[k0, i, :3, :3], Rn[k1]]))
for k in range(k0 + 1, k1): u = (t[k] - t0) / (t1 - t0); u = u * u * (3 - 2 * u); Rn[k] = sl(t0 + u * (t1 - t0)).as_matrix()
T[:, i, :3, :3] = Rn; T[:, i, :3, 3] = cw - np.einsum('kij,j->ki', Rn, c)
z['world_from_object'] = T; np.savez_compressed(dst / 'object_poses.npz', **z)
tr = dict(np.load(src / 'mini_sweeper_trajectory.npz')); Two = tr['T_world_obj'].copy(); Two[:, :3, :3] = Rn; Two[:, :3, 3] = T[:, i, :3, 3]
tr['T_world_obj'] = Two; tr['T_rgb_obj'] = np.linalg.inv(z['world_from_rgb']) @ Two; np.savez_compressed(dst / 'mini_sweeper_trajectory.npz', **tr)
tab = pq.read_table(src / 'reference.parquet'); row = tab.to_pylist()[0]; bn = list(row['object_body_names']); j = bn.index('mini_sweeper')
pos = np.array(row['object_body_position']); q = np.array(row['object_body_wxyz']); pos[:, j] = T[:, i, :3, 3]; qq = R.from_matrix(Rn).as_quat(); q[:, j] = qq[:, [3, 0, 1, 2]]
if bn[0] == row['object_name']:
    row['object_root_position'] = pos[:, 0].tolist(); rv = R.from_matrix(Rn).as_rotvec() if j == 0 else None
    if rv is not None: row['object_root_axis_angle'] = rv.tolist()
row['object_body_position'] = pos.tolist(); row['object_body_wxyz'] = q.tolist()
pq.write_table(pa.Table.from_pylist([row], schema=tab.schema), dst / 'reference.parquet')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
rel = np.degrees((R.from_matrix(Rn) * R.from_matrix(Rn[40]).inv()).magnitude())
info = dict(episode=30, object='mini_sweeper', window_s=[t0, t1], twin_axis_body=L.tolist(), twin_centre_body=c.tolist(), centre_trajectory_unchanged=True,
    evidence=dict(human_right_wrist_turn_deg_2_to_4s=30, original_track_turn_deg=178, twin_vs_rigid_hand_hold_deg='18-30 for 3.2-12 s', original_vs_rigid_hand_deg='150-166',
                  rgb_overlay='analysis/overlay_a.jpg: rot 0 and rot 180 both match the silhouette at 1 s and 4 s (symmetric object); +-90 do not'),
    corrected_turn_from_2s_deg={f'{s}': float(rel[int(s * 20)]) for s in (3.0, 3.4, 4.0, 8.0, 12.0)},
    original_object_poses_sha256=sha(src / 'object_poses.npz'), corrected_object_poses_sha256=sha(dst / 'object_poses.npz'),
    original_reference_parquet_sha256=sha(src / 'reference.parquet'), corrected_reference_parquet_sha256=sha(dst / 'reference.parquet'))
(dst / 'sweeper_track_correction.json').write_text(json.dumps(info, indent=2)); print(json.dumps(info['corrected_turn_from_2s_deg']), 'body origin max move cm', float(np.abs(T[:, i, :3, 3] - z['world_from_object'][:, i, :3, 3]).max()*100) if False else '')
