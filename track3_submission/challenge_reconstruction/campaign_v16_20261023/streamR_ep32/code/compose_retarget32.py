"""Stream R ep32 phase 6: re-target the whole reference (blue_cup + wooden_spoon) to the coworker's compose_eval tracks
(compose_target/episode_000032), expressed in OUR mesh body frames and OUR world.
 1. Body frames: our submitted meshes (v21 package blobs: cup 3c3c5328 = shortened 57bb recon cup, same body frame as 57bb; spoon c9ab88e3)
    are registered onto the compose meshes with the kit scorer's own M._register_object_body (seed 1_000_003*32+101*slot, exactly as
    score_vs_compose.py / the official scorer), giving G (pose of the compose body in our body frame); T_ours = T_compose @ inv(G).
    Cross-check with h2h meshreg.
 2. World: ONE rigid transform X (compose world = rectified-left camera at frame 0 -> our final_recon source world), Kabsch on the mesh
    surface points of both objects at frame 0 (our v21 reference vs compose). The compose camera/scale is adopted as given (no per-frame
    camera chain, no rescale). Residuals per object reported.
 3. Writes OUT/source_corrected/episode_000032 (final_recon copy, object_poses.npz both objects replaced, compose_retarget.json) and
    OUT/episode_corrected/episode_000032 (copy of BASE_EPD with manifest -> corrected source; tracking_reference / reference_trajectory
    object columns = setup_from_source @ new poses for frames >= 1; frame 0 kept = the physics scene initial state).
usage: compose_retarget32.py OUT_DIR [--base-epd DIR]"""
import sys, json, shutil, hashlib, base64, argparse, numpy as np, trimesh, pandas as pd
from pathlib import Path
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, '/data1/SRL_V2D/challenge_submissions_SRL/analysis_20261020/scripts'); import public_kit_score as K
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/h2h_spoon/code'); import meshreg
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--base-epd', default='/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/expanded_batch/repaired_export_inputs/episode_000032'); a = ap.parse_args()
out = Path(a.out).resolve(); OS, OE = out / 'source_corrected/episode_000032', out / 'episode_corrected/episode_000032'
if OS.exists() or OE.exists(): raise SystemExit('exists')
SRC = Path('/data1/SRL_V2D/challenge_reconstruction/final_recon/episode_000032'); CMP = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/compose_target/episode_000032')
EPD = Path(a.base_epd); V21 = '/data1/SRL_V2D/challenge_submissions/track3_v21_20261024/all_20_episodes_post/track3_submission.parquet'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
M = K.load_metric('MP-SR'); names = ['blue_cup', 'wooden_spoon']
s = pd.read_parquet(V21); blobs = {n: s[s.row_id == f'track_3/e000032/w000/f000000/o{i}'].object_mesh_b64.iloc[0] for i, n in enumerate(names)}
G = {}; reg = {}
for i, n in enumerate(names):
    m = trimesh.load(CMP / f'{n}_mesh_metric.obj', force='mesh'); m.merge_vertices()
    if len(m.faces) > 40000:
        try: m = m.simplify_quadric_decimation(face_count=40000)
        except Exception: pass
    m.remove_unreferenced_vertices(); v = np.asarray(m.vertices, np.float64); f = np.asarray(m.faces, np.int64); anchor, _ = M._surface_moments(v, f)
    cand = M._decode_object_mesh(blobs[n]); ca, rot = M._register_object_body(cand, (v, f), anchor, 1_000_003 * 32 + 101 * i)
    I = np.array([[0, 0, 0, 0, 0, 0, 1.]])[None, None]   # pose (p, q_xyzw?) -> use converter on identity
    conv = M._convert_object_body(np.array([[[0., 0, 0, 0, 0, 0, 1]]]), ca, rot)[0, 0]
    # conv = pose of the compose anchor frame expressed in our body: position, quaternion (kit order)
    q = conv[3:]; Ra = M._rotation_matrix(q[None])[0]; Gi = np.eye(4); Gi[:3, :3] = Ra; Gi[:3, 3] = conv[:3] - Ra @ anchor; G[n] = Gi
    cv = np.asarray(cand[0], float) if isinstance(cand, tuple) else np.asarray(cand.vertices, float)
    ours_pts = trimesh.Trimesh(cv, np.asarray(cand[1]) if isinstance(cand, tuple) else cand.faces).sample(5000)
    d = trimesh.proximity.closest_point(m, (ours_pts - Gi[:3, 3]) @ Gi[:3, :3])[1] if False else None
    h = meshreg.register(trimesh.Trimesh(cv, np.asarray(cand[1]) if isinstance(cand, tuple) else cand.faces), m)[0]; Gh = h[0]   # p_new = Gh p_ours
    Gh_pose = np.linalg.inv(Gh)   # pose of compose body in our body
    dR = np.degrees(R.from_matrix(Gh_pose[:3, :3].T @ Gi[:3, :3]).magnitude()); dt = np.linalg.norm(Gh_pose[:3, 3] - Gi[:3, 3]) * 100
    reg[n] = dict(kit_G=Gi.tolist(), h2h_chamfer_mm=float(h[1] * 1000), kit_vs_h2h_deg=float(dR), kit_vs_h2h_cm=float(dt), compose_anchor=anchor.tolist())
    print(n, json.dumps({k: (round(v, 3) if isinstance(v, float) else '') for k, v in reg[n].items() if k != 'kit_G' and k != 'compose_anchor'}))
z = dict(np.load(SRC / 'object_poses.npz', allow_pickle=False)); zn = [str(x) for x in z['names']]; W0 = z['world_from_object'].copy(); N = len(W0)
Tc = {n: np.load(CMP / f'{n}_trajectory.npz')['T_world_obj'].astype(np.float64) for n in names}
To = {n: Tc[n] @ np.linalg.inv(G[n]) for n in names}   # our body frame, compose world
pts = {n: trimesh.load(SRC / f'assets/{n}.obj', force='mesh').sample(4000) for n in names}
A = np.vstack([pts[n] @ To[n][0, :3, :3].T + To[n][0, :3, 3] for n in names]); B = np.vstack([pts[n] @ W0[0, zn.index(n), :3, :3].T + W0[0, zn.index(n), :3, 3] for n in names])
ca_, cb_ = A.mean(0), B.mean(0); U, S_, Vt = np.linalg.svd((A - ca_).T @ (B - cb_)); Rx = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
X = np.eye(4); X[:3, :3] = Rx; X[:3, 3] = cb_ - Rx @ ca_
W = W0.copy()
for n in names: W[:, zn.index(n)] = X @ To[n]
anc = {n: trimesh.load(SRC / f'assets/{n}.obj', force='mesh', process=False).centroid for n in names}
def diff(Ta, Tb, n): return float(np.linalg.norm((Ta[..., :3, :3] @ anc[n] + Ta[..., :3, 3]) - (Tb[..., :3, :3] @ anc[n] + Tb[..., :3, 3]), axis=-1).mean() * 100), float(np.degrees(np.mean([R.from_matrix(x.T @ y).magnitude() for x, y in zip(Ta[..., :3, :3].reshape(-1, 3, 3), Tb[..., :3, :3].reshape(-1, 3, 3))])))
f0 = {n: diff(W[0:1, zn.index(n)], W0[0:1, zn.index(n)], n) for n in names}; allf = {n: diff(W[:, zn.index(n)], W0[:, zn.index(n)], n) for n in names}
Sm = np.asarray(json.loads((EPD / 'input_manifest.json').read_text())['setup_from_source'])
bottoms = {n: [float((trimesh.load(SRC / f'assets/{n}.obj', force='mesh').vertices @ (Sm @ T)[:3, :3].T + (Sm @ T)[:3, 3])[:, 2].min() * 100) for T in (W[0, zn.index(n)], W[-1, zn.index(n)], W0[0, zn.index(n)], W0[-1, zn.index(n)])] for n in names}
print('frame-0 residual (new vs ours) cm/deg', {k: tuple(round(x, 2) for x in v) for k, v in f0.items()}, '| whole-clip shift', {k: tuple(round(x, 2) for x in v) for k, v in allf.items()})
print('mesh bottom z in setup frame (cm): new f0, new end, ours f0, ours end', {k: [round(x, 2) for x in v] for k, v in bottoms.items()})
shutil.copytree(SRC, OS, symlinks=True); z['world_from_object'] = W; (OS / 'object_poses.npz').unlink(); np.savez_compressed(OS / 'object_poses.npz', **z)
for n in names:
    p = OS / f'{n}_trajectory.npz'
    if p.exists():
        tr = dict(np.load(SRC / f'{n}_trajectory.npz', allow_pickle=False))
        for k in list(tr):
            if tr[k].ndim == 3 and tr[k].shape[-2:] == (4, 4) and len(tr[k]) == N:
                if np.allclose(tr[k], W0[:, zn.index(n)], atol=1e-9): tr[k] = W[:, zn.index(n)]
                elif np.allclose(tr[k], np.linalg.inv(z['world_from_rgb']) @ W0[:, zn.index(n)], atol=1e-9): tr[k] = np.linalg.inv(z['world_from_rgb']) @ W[:, zn.index(n)]
        p.unlink(); np.savez_compressed(p, **tr)
rec = dict(episode=32, method='compose tracks -> our body frames via the kit scorer registration (M._register_object_body) -> our world via ONE rigid transform (Kabsch, frame-0 mesh points of both objects)',
           compose_dir=str(CMP), compose_track_sha256={n: sha(CMP / f'{n}_trajectory.npz') for n in names}, compose_mesh_sha256={n: sha(CMP / f'{n}_mesh_metric.obj') for n in names},
           our_submitted_mesh_from='v21 package blobs (cup 3c3c5328, spoon c9ab88e3)', registration=reg, world_transform_X=X.tolist(),
           frame0_residual_cm_deg=f0, whole_clip_shift_vs_v21_reference_cm_deg=allf, per_frame_shift_cm={n: (np.linalg.norm((W[:, zn.index(n), :3, :3] @ anc[n] + W[:, zn.index(n), :3, 3]) - (W0[:, zn.index(n), :3, :3] @ anc[n] + W0[:, zn.index(n), :3, 3]), axis=1) * 100).round(3).tolist() for n in names},
           mesh_bottom_z_setup_cm=bottoms, original_object_poses_sha256=sha(SRC / 'object_poses.npz'), corrected_object_poses_sha256=sha(OS / 'object_poses.npz'), script=str(Path(__file__).resolve()), script_sha256=sha(__file__))
(OS / 'compose_retarget.json').write_text(json.dumps(rec, indent=1))
OE.mkdir(parents=True)
for p in EPD.iterdir(): shutil.copy2(p, OE / p.name)
m = json.loads((EPD / 'input_manifest.json').read_text())
trk = dict(np.load(EPD / 'tracking_reference.npz', allow_pickle=False)); q = trk['qpos'].copy(); cols = {}
for n in names:
    TS0, TS = Sm @ W0[:, zn.index(n)], Sm @ W[:, zn.index(n)]; q0 = R.from_matrix(TS0[:, :3, :3]).as_quat()[:, [3, 0, 1, 2]]
    adr = next(c for c in range(q.shape[1] - 6) if np.abs(q[:, c:c + 3] - TS0[:, :3, 3]).max() < 1e-6 and np.minimum(np.abs(q[:, c + 3:c + 7] - q0).max(1), np.abs(q[:, c + 3:c + 7] + q0).max(1)).max() < 1e-6)
    qn = R.from_matrix(TS[:, :3, :3]).as_quat()[:, [3, 0, 1, 2]]; qn *= np.sign(np.sum(qn * q[:, adr + 3:adr + 7], 1))[:, None]
    q[1:, adr:adr + 3] = TS[1:, :3, 3]; q[1:, adr + 3:adr + 7] = qn[1:]; cols[n] = [adr, adr + 7]
trk['qpos'] = q; np.savez(OE / 'tracking_reference.npz', **trk)
rt = dict(np.load(EPD / 'reference_trajectory.npz', allow_pickle=False))
if rt['qpos'].shape == q.shape: rt['qpos'] = rt['qpos'].copy(); rt['qpos'][1:] = np.where(np.isin(np.arange(q.shape[1]), [c for n in names for c in range(*cols[n])])[None], q[1:], rt['qpos'][1:])
for i, n in enumerate([str(x) for x in rt['object_names']]):
    TS = Sm @ W[:, zn.index(n)]; rt['object_positions'] = rt['object_positions'].copy(); rt['object_quaternions_wxyz'] = rt['object_quaternions_wxyz'].copy()
    rt['object_positions'][1:, i] = TS[1:, :3, 3]; qq = R.from_matrix(TS[:, :3, :3]).as_quat()[:, [3, 0, 1, 2]]; rt['object_quaternions_wxyz'][1:, i] = qq[1:]
np.savez(OE / 'reference_trajectory.npz', **rt)
m.update(source=str(OS), source_reference=str(OS / 'reference.parquet'), source_reference_sha256=sha(OS / 'reference.parquet'), source_object_poses_sha256=sha(OS / 'object_poses.npz'))
m['compose_retarget'] = dict(record=str(OS / 'compose_retarget.json'), record_sha256=sha(OS / 'compose_retarget.json'), script=str(Path(__file__).resolve()), script_sha256=sha(__file__), base_episode_dir=str(EPD),
    base_manifest_sha256=sha(EPD / 'input_manifest.json'), tracking_reference_object_columns=cols, frame0_kept='tracking_reference frame 0 = physics scene initial state (original); scoring source frame 0 = compose')
(OE / 'input_manifest.json').write_text(json.dumps(m, indent=2) + '\n'); print('wrote', OS, OE)
