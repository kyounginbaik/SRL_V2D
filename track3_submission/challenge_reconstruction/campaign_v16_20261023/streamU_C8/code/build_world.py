"""Stream U (phase 6, ep8): build the compose-target episode world.
 * source_compose/episode_000008   : SCORING source = copy of final_recon/episode_000008 whose object_poses.npz world_from_object is
                                     the compose track converted to our mesh bodies + our source world (compose_reference.py);
                                     reference.parquet is the unchanged original (evaluator reads only object_poses + manifest).
 * source_physical/episode_000008  : ADAPTER input = same, but resting frames projected onto the rebuilt table plane (the physical
                                     rest pose of OUR mesh: rest axis || plane normal, lowest point on the plane) and carried frames
                                     shifted by the blended rest corrections; reference.parquet object columns rewritten to match.
                                     It defines the scene's frame-0 object poses and tracking_reference (teacher/ILC targets).
 * world/episodes/episode_000008   : unchanged adapter (prepare_episodes.prepare) with setup_from_source = plane frame, then
                                     input_manifest.source -> source_compose (scored against the compose poses).
Table plane: normal = the white_pot rest axis under its compose frame-0 pose (pot = the scorer's rotation anchor, so its frame-0
orientation is reproduced exactly); height = pot support. 6.5 deg from the streamB stereo plane; the compose cup rest axis agrees
within 2.2 deg. usage: build_world.py REFERENCE_DIR OUT_DIR"""
import sys, json, hashlib, shutil, os
from pathlib import Path
import numpy as np, trimesh, pyarrow as pa, pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R
CR = Path('/data1/SRL_V2D/challenge_reconstruction'); LEG = CR / 'dexagent_final_recon_20261017'; SRC = CR / 'final_recon/episode_000008'
CALIB = CR / 'campaign_v16_20261023/streamB_tilt38/ep8/calibrated_inputs/episodes/episode_000008'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
REF, OUT = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
if OUT.exists(): raise SystemExit('exists ' + str(OUT))
S_old = np.array(json.loads((CALIB / 'input_manifest.json').read_text())['setup_from_source'])
src = np.load(SRC / 'object_poses.npz'); names = [str(n) for n in src['names']]; ours = src['world_from_object'].astype(float)
comp = np.load(REF / 'compose_in_source_world.npz')['world_from_object'].astype(float); assert [str(n) for n in np.load(REF / 'compose_in_source_world.npz')['names']] == names
T = len(comp); V = [np.asarray(trimesh.load(SRC / f'assets/{n}.obj', force='mesh').vertices) for n in names]
UP = [-np.array([0.005, -0.005, 1.]), -np.array([-0.986, -0.022, -0.167])]; UP = [u / np.linalg.norm(u) for u in UP]   # convex-hull support facets (cup bottom disc, pot flat bottom), body frame
L = np.einsum('ij,tkjl->tkil', S_old, comp)
n = L[0, 1, :3, :3] @ UP[1]; n /= np.linalg.norm(n)
h = float((V[1] @ L[0, 1, :3, :3].T + L[0, 1, :3, 3]) @ n).min() if False else float(((V[1] @ L[0, 1, :3, :3].T + L[0, 1, :3, 3]) @ n).min())
z = L[:, :, 2, 3]; rest = (np.abs(z - z[:1]) < .01) | (np.abs(z - z[-1:]) < .01)
win = {}
for k in range(2):
    r = rest[:, k]; a = int(np.argmin(r)); b = int(len(r) - np.argmin(r[::-1])); m = np.zeros_like(r); m[:a] = True; m[b:] = True; rest[:, k] = m; win[names[k]] = dict(lift_frame=a, setdown_frame=b)
def project(Tm, k):
    up = Tm[:3, :3] @ UP[k]; Rm = R.align_vectors([n], [up])[0].as_matrix(); pts = V[k] @ Tm[:3, :3].T + Tm[:3, 3]; lo = pts[np.argmin(pts @ n)]
    T2 = Tm.copy(); T2[:3, :3] = Rm @ Tm[:3, :3]; T2[:3, 3] = Rm @ (Tm[:3, 3] - lo) + lo; pts2 = V[k] @ T2[:3, :3].T + T2[:3, 3]; T2[:3, 3] += n * (h - (pts2 @ n).min()); return T2
P = L.copy()
for k in range(2):
    a, b = win[names[k]]['lift_frame'], win[names[k]]['setdown_frame']
    for t in np.flatnonzero(rest[:, k]): P[t, k] = project(L[t, k], k)
    Da = P[a - 1, k, :3, 3] - L[a - 1, k, :3, 3]; Db = P[b, k, :3, 3] - L[b, k, :3, 3]
    for t in range(a, b): w = (t - a) / max(1, b - a); P[t, k, :3, 3] = L[t, k, :3, 3] + Da * (1 - w) + Db * w
phys = np.einsum('ij,tkjl->tkil', np.linalg.inv(S_old), P)
# rigidity / orthonormality cleanup (float roundoff)
for X in (comp, phys):
    U_, _, Vt = np.linalg.svd(X[..., :3, :3]); X[..., :3, :3] = U_ @ Vt
Rn = R.align_vectors([[0, 0, 1.]], [n])[0].as_matrix(); G = np.eye(4); G[:3, :3] = Rn; G[2, 3] = -h; S_new = G @ S_old
OUT.mkdir(parents=True)
def make_source(dst, poses, rewrite_parquet):
    dst.mkdir(parents=True)
    for f in ('config.json', 'manifest.json', 'camera_motion.npz', 'reference.parquet'): shutil.copy2(SRC / f, dst / f)
    for d in ('assets', 'stereo_calibration'): shutil.copytree(SRC / d, dst / d)
    np.savez(dst / 'object_poses.npz', world_from_object=poses, world_from_rgb=src['world_from_rgb'], frames=src['frames'], names=src['names'])
    if rewrite_parquet:
        t = pq.read_table(dst / 'reference.parquet'); rows = t.to_pylist(); assert len(rows) == 1
        rows[0]['object_body_position'] = poses[:, :, :3, 3].tolist()
        rows[0]['object_body_wxyz'] = R.from_matrix(poses[..., :3, :3].reshape(-1, 3, 3)).as_quat(scalar_first=True).reshape(T, len(names), 4).tolist()
        pq.write_table(pa.Table.from_pylist(rows, schema=t.schema).replace_schema_metadata(t.schema.metadata), dst / 'reference.parquet')
SC = OUT / 'source_compose/episode_000008'; SP = OUT / 'source_physical/episode_000008'
make_source(SC, comp, False); make_source(SP, phys, True)
sys.path[:0] = [str(LEG / 'code'), '/data1/SRL_V2D/third_party/DexAgent']
import prepare_episodes as adapter
from evaluate_mp_sr import geometry_from_reference
fake = OUT / 'adapter_root'; (fake / 'challenge_reconstruction/final_recon').mkdir(parents=True); os.symlink(SP, fake / 'challenge_reconstruction/final_recon/episode_000008')
adapter.ROOT = fake; adapter.WORKSPACE = OUT / 'world'
plane = dict(normal_in_streamB_level_frame=n.tolist(), height_in_streamB_level_frame_m=h, tilt_from_streamB_plane_deg=float(np.degrees(np.arccos(n[2]))),
             derivation='normal = white_pot rest axis (convex-hull bottom facet, body ' + str(np.round(-UP[1], 3).tolist()) + ') under its compose frame-0 pose; height = pot support',
             cup_rest_axis_vs_plane_deg=float(np.degrees(np.arccos(np.clip((L[0, 0, :3, :3] @ UP[0]) @ n, -1, 1)))),
             cup_bottom_below_plane_cm_frame0=float((h - ((V[0] @ L[0, 0, :3, :3].T + L[0, 0, :3, 3]) @ n).min()) * 100),
             streamB_setup_from_source=S_old.tolist(), setup_from_source=S_new.tolist(), rest_windows=win)
(OUT / 'table_plane.json').write_text(json.dumps(plane, indent=1) + '\n')
evidence = dict(path=str(OUT / 'table_plane.json'), sha256=sha(OUT / 'table_plane.json'), method='table plane rebuilt from the compose resting poses (phase 6 recipe); see build_world.py docstring')
manifest = adapter.prepare(8, S_new, evidence)
ep = OUT / 'world/episodes/episode_000008'
(ep / 'metric_geometry.json').write_text(json.dumps(geometry_from_reference(SC), indent=2) + '\n')
ref = np.load(ep / 'reference_trajectory.npz'); Tt = np.broadcast_to(np.eye(4), phys.shape).copy()
Tt[:, :, :3, :3] = R.from_quat(ref['object_quaternions_wxyz'].reshape(-1, 4), scalar_first=True).as_matrix().reshape(phys.shape[:2] + (3, 3)); Tt[:, :, :3, 3] = ref['object_positions']
rt = float(abs(np.linalg.inv(S_new) @ Tt - phys).max()); assert rt < 1e-6, rt
m = json.loads((ep / 'input_manifest.json').read_text())
m['physical_init_source'] = dict(source=m['source'], object_poses_sha256=m['source_object_poses_sha256'], reference=m['source_reference'], reference_sha256=m['source_reference_sha256'],
    note='scene frame-0 poses + tracking_reference objects (teacher/ILC targets) = compose poses with resting frames projected onto the rebuilt plane')
m['source'] = str(SC); m['source_reference'] = str(SC / 'reference.parquet'); m['source_reference_sha256'] = sha(SC / 'reference.parquet'); m['source_object_poses_sha256'] = sha(SC / 'object_poses.npz')
m['compose_target'] = dict(reference_dir=str(REF), diagnostics=str(REF / 'diagnostics.json'), diagnostics_sha256=sha(REF / 'diagnostics.json'), table_plane=str(OUT / 'table_plane.json'),
    note='scoring reference = coworker compose_eval track (compose_target/episode_000008) converted into our mesh bodies by the kit registration and into our source world by one rigid camera-chain transform')
(ep / 'input_manifest.json').write_text(json.dumps(m, indent=2) + '\n')
# lineage: per-frame shift vs the v21 reference (final_recon)
lin = dict(schema='streamU_compose_reference_lineage_v1', scoring_source=str(SC), scoring_object_poses_sha256=sha(SC / 'object_poses.npz'), physical_source=str(SP), physical_object_poses_sha256=sha(SP / 'object_poses.npz'),
           v21_reference=str(SRC / 'object_poses.npz'), v21_reference_sha256=sha(SRC / 'object_poses.npz'), compose_inputs=json.loads((REF / 'diagnostics.json').read_text())['inputs'],
           W_source_from_compose=np.load(REF / 'compose_in_source_world.npz')['W'].tolist(), table_plane=plane, roundtrip_max_error=rt, shift_vs_v21={})
for k, nm in enumerate(names):
    for lab, X in (('scoring', comp), ('physical', phys)):
        dp = np.linalg.norm(X[:, k, :3, 3] - ours[:, k, :3, 3], axis=1) * 100; dr = np.degrees(R.from_matrix(np.einsum('tji,tjk->tik', ours[:, k, :3, :3], X[:, k, :3, :3])).magnitude())
        lin['shift_vs_v21'][f'{nm}/{lab}'] = dict(cm_mean=float(dp.mean()), cm_max=float(dp.max()), deg_mean=float(dr.mean()), deg_max=float(dr.max()), cm_per_frame=np.round(dp, 3).tolist())
    dp = np.linalg.norm(phys[:, k, :3, 3] - comp[:, k, :3, 3], axis=1) * 100; lin['physical_vs_scoring_' + nm] = dict(cm_mean=float(dp.mean()), cm_max=float(dp.max()))
(OUT / 'lineage.json').write_text(json.dumps(lin, indent=1) + '\n')
print(json.dumps(dict(episode_dir=str(ep), plane={k: v for k, v in plane.items() if 'setup' not in k}, roundtrip=rt, shift={k: {kk: round(vv, 2) for kk, vv in v.items() if kk != 'cm_per_frame'} for k, v in lin['shift_vs_v21'].items()},
                      phys_vs_scoring={nm: lin['physical_vs_scoring_' + nm] for nm in names}), indent=1))
