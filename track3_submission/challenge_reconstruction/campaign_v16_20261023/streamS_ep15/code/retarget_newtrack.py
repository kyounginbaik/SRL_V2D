"""Stream S (ep15, phase 5 follow-up): re-targeted reference with the coworker's NEW wooden_spoon track (compose_eval_20261021,
h2h_spoon verdict NEW/high) in OUR world frame. The rack track and every other source file stay byte-identical.

Chain (shared camera chain; h2hlib.Ep.cam_T): camera-a pose of a NEW object at frame f = Ta @ inv(cam0_T_cam[f]) @ T_world_obj[f];
OUR world pose = world_from_rgb[f] @ (camera-a pose)  (our ep15 reference already lives in the official cam-a frame, h2hlib.model_of).
A_f = world_from_rgb[f] @ Ta @ inv(cam0_T_cam[f]) is checked for constancy; the median constant A is used (no per-frame camera noise)
if the spread is tiny, else per-frame A_f. Verification: NEW rack (static) vs OUR rack; NEW vs OUR spoon on all frames (stats).
Mesh frames: OUR spoon mesh 6cc4d2b2... vs their wooden_spoon_mesh_metric.obj registered (h2h meshreg.register: p_new = G p_ours);
OUR mesh is kept and the poses are converted: W_ours_body = W_new_body @ G.
usage: retarget_newtrack.py OUT_SOURCE_DIR REPORT_JSON [--no-mesh-reg]"""
import hashlib, json, shutil, sys
from pathlib import Path
import numpy as np, trimesh
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/h2h_spoon/code')
import meshreg
R7 = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019'); SRC = R7 / 'hawor_stereo_recon_v2/episode_000015'
NEW = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/h2h_spoon/inputs_new/episode_000015')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rotdeg(Ra, Rb): return np.degrees(np.linalg.norm(R.from_matrix(np.einsum('...ji,...jk->...ik', Ra, Rb)).as_rotvec(), axis=-1))
def stats(x): x = np.asarray(x); return dict(mean=float(x.mean()), median=float(np.median(x)), p90=float(np.percentile(x, 90)), max=float(x.max()), min=float(x.min()))
out, rep_p = Path(sys.argv[1]), Path(sys.argv[2]); use_reg = '--no-mesh-reg' not in sys.argv
if out.exists(): raise SystemExit(f'exists: {out}')
z = np.load(SRC / 'object_poses.npz'); names = [str(n) for n in z['names']]; W = z['world_from_object'].astype(np.float64); wfc = z['world_from_rgb'].astype(np.float64)
cm = np.load(NEW / 'camera_motion.npz'); c0Tc = cm['cam0_T_cam'].astype(np.float64); Ta = cm['T_a_from_rectL'].astype(np.float64)
Tn = {o: np.load(NEW / f'{o}_trajectory.npz')['T_world_obj'].astype(np.float64) for o in names}; N = len(W); assert all(len(v) == N for v in Tn.values()) and len(c0Tc) == N
Af = np.stack([wfc[f] @ Ta @ np.linalg.inv(c0Tc[f]) for f in range(N)])
A = Af[0].copy(); rots = R.from_matrix(Af[:, :3, :3]); A[:3, :3] = rots.mean().as_matrix(); A[:3, 3] = np.median(Af[:, :3, 3], 0)
spread = dict(pos_cm=stats(np.linalg.norm(Af[:, :3, 3] - A[:3, 3], axis=1) * 100), rot_deg=stats(rotdeg(Af[:, :3, :3], A[:3, :3][None])))
const = spread['pos_cm']['max'] < 0.1 and spread['rot_deg']['max'] < 0.1
Wn = {o: (np.einsum('ij,njk->nik', A, Tn[o]) if const else np.einsum('nij,njk->nik', Af, Tn[o])) for o in names}
# mesh registration (our body frame -> their body frame)
mo = trimesh.load(SRC / 'assets/wooden_spoon_visual.obj', force='mesh', process=False); mn = trimesh.load(NEW / 'wooden_spoon_mesh_metric.obj', force='mesh', process=False)
reg = meshreg.register(mo, mn); G, cham = reg[0]; G2, cham2 = reg[1]
greg = dict(our_mesh=str(SRC / 'assets/wooden_spoon_visual.obj'), our_sha256=sha(SRC / 'assets/wooden_spoon_visual.obj'), new_mesh=str(NEW / 'wooden_spoon_mesh_metric.obj'),
            new_sha256=sha(NEW / 'wooden_spoon_mesh_metric.obj'), identical=sha(SRC / 'assets/wooden_spoon_visual.obj') == sha(NEW / 'wooden_spoon_mesh_metric.obj'),
            G_new_from_ours=G, chamfer_mm=cham * 1000, G_translation_cm=G[:3, 3] * 100, G_rotation_deg=float(rotdeg(G[:3, :3], np.eye(3))),
            second_best_chamfer_mm=cham2 * 1000, second_best_rotation_deg=float(rotdeg(G2[:3, :3], np.eye(3))),
            our_extents_m=mo.extents, new_extents_m=mn.extents, applied=bool(use_reg))
Gu = G if use_reg else np.eye(4)
si, ri = names.index('wooden_spoon'), names.index('plastic_dish_rack')
Wspoon = np.einsum('nij,jk->nik', Wn['wooden_spoon'], Gu)            # pose of OUR mesh body frame on the NEW track
# verification
pts = np.asarray(trimesh.sample.sample_surface(mo, 4000, seed=0)[0])
def pdiff(Wa, Wb, P): Xa = np.einsum('nij,kj->nki', Wa[:, :3, :3], P) + Wa[:, None, :3, 3]; Xb = np.einsum('nij,kj->nki', Wb[:, :3, :3], P) + Wb[:, None, :3, 3]; return np.linalg.norm(Xa - Xb, axis=2).mean(1) * 100
mr = trimesh.load(SRC / 'assets/plastic_dish_rack_visual.obj', force='mesh', process=False); pr = np.asarray(trimesh.sample.sample_surface(mr, 4000, seed=0)[0])
rack_d = pdiff(Wn['plastic_dish_rack'], W[:, ri], pr); rack_r = rotdeg(Wn['plastic_dish_rack'][:, :3, :3], W[:, ri, :3, :3])
sp_d = pdiff(Wspoon, W[:, si], pts); sp_r = rotdeg(Wspoon[:, :3, :3], W[:, si, :3, :3]); dpos = np.linalg.norm(Wspoon[:, :3, 3] - W[:, si, :3, 3], axis=1) * 100
# displacement expressed along the spoon long axis (largest extent axis of our mesh, body frame of the OLD pose)
ax = int(np.argmax(mo.extents)); along = np.einsum('ni,ni->n', Wspoon[:, :3, 3] - W[:, si, :3, 3], W[:, si, :3, ax]) * 100
agree = np.flatnonzero(sp_d < np.percentile(sp_d, 10))
# write the corrected source copy
shutil.copytree(SRC, out, ignore=shutil.ignore_patterns('archive_wilor', 'archive_before_stereo', '*.mp4', '__pycache__', '*.pyc'))
W2 = z['world_from_object'].copy(); W2[:, si] = Wspoon.astype(W2.dtype)
np.savez(out / 'object_poses.npz', **{k: (W2 if k == 'world_from_object' else z[k]) for k in z.files})
t = np.load(SRC / 'wooden_spoon_trajectory.npz'); tt = {k: t[k] for k in t.files}
tt['T_world_obj'] = Wspoon.astype(t['T_world_obj'].dtype); tt['T_rgb_obj'] = np.einsum('nij,njk->nik', np.linalg.inv(wfc), Wspoon).astype(t['T_rgb_obj'].dtype)
np.savez(out / 'wooden_spoon_trajectory.npz', **tt)
import pyarrow as pa, pyarrow.parquet as pq
tab = pq.read_table(SRC / 'reference.parquet'); row = tab.to_pylist()[0]; assert len(tab) == 1
bp = np.array(row['object_body_position']); bq = np.array(row['object_body_wxyz']); bn = list(row['object_body_names']); bi = bn.index('wooden_spoon')
q_old = R.from_matrix(W[:, si, :3, :3]).as_quat()[:, [3, 0, 1, 2]]
assert np.allclose(bp[:, bi], W[:, si, :3, 3], atol=1e-6) and np.allclose(np.abs(np.einsum('ni,ni->n', bq[:, bi], q_old)), 1, atol=1e-6), 'parquet != object_poses (old)'
qn = R.from_matrix(Wspoon[:, :3, :3]).as_quat()[:, [3, 0, 1, 2]]; qn *= np.sign(np.einsum('ni,ni->n', qn, bq[:, bi]))[:, None]
bp[:, bi] = Wspoon[:, :3, 3]; bq[:, bi] = qn
cols = {n: tab.column(n) for n in tab.column_names}
cols['object_body_position'] = pa.array([bp.tolist()], type=tab.schema.field('object_body_position').type)
cols['object_body_wxyz'] = pa.array([bq.tolist()], type=tab.schema.field('object_body_wxyz').type)
pq.write_table(pa.table(cols, schema=tab.schema), out / 'reference.parquet')
(out / 'STREAM_S_NEWTRACK.txt').write_text(f'Copy of {SRC}; object_poses.npz world_from_object[:, wooden_spoon] and wooden_spoon_trajectory.npz replaced by the NEW compose_eval_20261021 '
    f'spoon track ({NEW}/wooden_spoon_trajectory.npz sha256 {sha(NEW / "wooden_spoon_trajectory.npz")}) mapped through the shared camera chain and the fixed mesh registration; '
    f'reference.parquet object_body_position/wxyz of the spoon updated to match; everything else byte-identical. Report: {rep_p}\n')
rep = dict(script=str(Path(__file__).resolve()), script_sha256=sha(__file__), source_dir=str(SRC), source_object_poses_sha256=sha(SRC / 'object_poses.npz'),
           new_track=str(NEW / 'wooden_spoon_trajectory.npz'), new_track_sha256=sha(NEW / 'wooden_spoon_trajectory.npz'), new_camera_motion_sha256=sha(NEW / 'camera_motion.npz'),
           new_rack_track_sha256=sha(NEW / 'plastic_dish_rack_trajectory.npz'), output_dir=str(out), output_object_poses_sha256=sha(out / 'object_poses.npz'),
           output_spoon_trajectory_sha256=sha(out / 'wooden_spoon_trajectory.npz'), output_reference_parquet_sha256=sha(out / 'reference.parquet'), source_reference_parquet_sha256=sha(SRC / 'reference.parquet'),
           parquet_note='only object_body_position / object_body_wxyz of wooden_spoon replaced; hand/MANO/robot columns (incl. MANO object-contact points of the old track) unchanged',
           chain='W_ours_body[f] = A @ T_world_obj_new[f] @ G ; A = world_from_rgb[f] @ T_a_from_rectL @ inv(cam0_T_cam[f]) (constant: %s)' % const,
           A_world_from_newworld=A, A_frame_spread=spread, A_constant_used=const, mesh_registration=greg,
           verification=dict(rack_new_vs_ours_cm=stats(rack_d), rack_new_vs_ours_deg=stats(rack_r), rack_kept='ours (static), unchanged',
                             spoon_new_vs_ours_mean_point_cm=stats(sp_d), spoon_center_shift_cm=stats(dpos), spoon_rot_deg=stats(sp_r), spoon_shift_along_handle_cm=stats(along),
                             spoon_best_agreeing_10pct_frames=agree, spoon_residual_on_agreeing_frames_cm=stats(sp_d[agree]), spoon_rot_on_agreeing_frames_deg=stats(sp_r[agree])),
           per_frame=dict(spoon_mean_point_cm=sp_d, spoon_center_shift_cm=dpos, spoon_rot_deg=sp_r, along_handle_cm=along, rack_cm=rack_d),
           per_second=[dict(t_s=s / 20, spoon_cm=float(sp_d[s:s + 20].mean()), spoon_deg=float(sp_r[s:s + 20].mean()), along_cm=float(along[s:s + 20].mean())) for s in range(0, N, 20)])
rep_p.write_text(json.dumps(rep, indent=1, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else (bool(x) if isinstance(x, np.bool_) else str(x))) + '\n')
v = rep['verification']; print(json.dumps(dict(A_spread=spread, const=const, mesh=dict(chamfer_mm=greg['chamfer_mm'], t_cm=greg['G_translation_cm'].round(3).tolist(), rot=greg['G_rotation_deg'], second=greg['second_best_chamfer_mm']),
      rack_cm=v['rack_new_vs_ours_cm'], rack_deg=v['rack_new_vs_ours_deg'], spoon_cm=v['spoon_new_vs_ours_mean_point_cm'], spoon_deg=v['spoon_rot_deg'], along=v['spoon_shift_along_handle_cm'],
      agree_cm=v['spoon_residual_on_agreeing_frames_cm'], per_second=rep['per_second']), indent=1, default=str))
