"""Stream V (ep19, v21C): build (a) a corrected-source copy of final_recon/episode_000019 whose object targets are the coworker's
compose_eval poses expressed in OUR mesh body frames (scorer-exact registration) and in OUR world by one rigid transform X,
(b) a corrected-track episode dir (copy of the repaired export adapter, reference columns + manifest updated), and
(c) a physics scene that differs from the shipped frozen scene ONLY in the 'home' keyframe object poses (= new frame-0 targets).
X = dust-pan frame-0 match (static scene: the compose dust pan then rests on our table at the start (0.12 cm) and at the end
(0.02 cm) exactly like ours; the compose sweeper frame 0 also touches our table, lowest point -0.01 cm).
The mini_sweeper keeps the compose face (= the face of our un-relabelled source track; NO half-turn relabel G).
New directories only."""
import json, shutil, hashlib, sys, re
import numpy as np, mujoco
from pathlib import Path
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamV_C19/code')
from common19 import *
import argparse
ap = argparse.ArgumentParser(); ap.add_argument('--x', default='dp_f0', help='dp_f0 | sw_f0 | mix:ALPHA (rotation slerp / translation lerp from sw_f0 (0) to dp_f0 (1))'); ap.add_argument('--suffix', default=''); ap.add_argument('--keep-home', action='append', default=[], help='object whose home keyframe stays the shipped (physically settled) pose; its frame-0 target is unchanged'); ap.add_argument('--clear-mm', type=float, default=None, help='raise the home keyframe of any object whose mesh dips below z=0 so its lowest point is at +clear (keyframe only; targets unchanged)')
A = ap.parse_args()
OUT_SRC = V / f'staging_C/source_compose{A.suffix}/episode_000019'; OUT_EP = V / f'staging_C/episode_compose{A.suffix}/episode_000019'; OUT_SC = V / f'staging_C/scene{A.suffix}'
for p in (OUT_SRC, OUT_EP, OUT_SC):
    if p.exists(): raise SystemExit('exists ' + str(p))
man = json.loads((EPDIR / 'input_manifest.json').read_text()); S = np.array(man['setup_from_source']); Sinv = np.array(man['source_from_setup'])
assert man['source'] == str(SRC) and man['source_object_poses_sha256'] == sha(SRC / 'object_poses.npz')
packed = packed_meshes(); reg = registration(packed); comp = compose_tracks()
d = np.load(SRC / 'object_poses.npz'); W = d['world_from_object'].astype(float); names = [str(x) for x in d['names']]; assert names == NAMES
ours = {n: np.einsum('ij,tjk->tik', S, W[:, i]) for i, n in enumerate(NAMES)}
cb = {n: comp[n] @ reg[n]['M'] for n in NAMES}
Xof = lambda k: ours[k][0] @ np.linalg.inv(cb[k][0])
if A.x.startswith('mix:'):
    al = float(A.x[4:]); Xs, Xd = Xof('mini_sweeper'), Xof('mini_dust_pan'); X = np.eye(4)
    X[:3, :3] = (R.from_matrix(Xs[:3, :3]) * R.from_rotvec(al * (R.from_matrix(Xs[:3, :3]).inv() * R.from_matrix(Xd[:3, :3])).as_rotvec())).as_matrix()
    c0 = 0.5 * (cb['mini_sweeper'][0][:3, 3] + cb['mini_dust_pan'][0][:3, 3]); X[:3, 3] = (1 - al) * (Xs[:3, :3] @ c0 + Xs[:3, 3]) + al * (Xd[:3, :3] @ c0 + Xd[:3, 3]) - X[:3, :3] @ c0
else: X = Xof({'dp_f0': 'mini_dust_pan', 'sw_f0': 'mini_sweeper'}[A.x])
new_setup = {n: np.einsum('ij,tjk->tik', X, cb[n]) for n in NAMES}
for n in NAMES: new_setup[n][:, :3, :3] = R.from_matrix(new_setup[n][:, :3, :3]).as_matrix()   # compose rotations re-orthonormalised (as the scorer's quaternion conversion does)
Wn = np.stack([np.einsum('ij,tjk->tik', Sinv, new_setup[n]) for n in NAMES], 1)
for n in NAMES:   # rigidity / round trip
    Rm = new_setup[n][:, :3, :3]; assert np.allclose(np.swapaxes(Rm, 1, 2) @ Rm, np.eye(3), atol=1e-12)
# (a) corrected source
shutil.copytree(SRC, OUT_SRC, ignore=shutil.ignore_patterns('*.mp4', 'audit_before'))
(OUT_SRC / 'object_poses.npz').unlink()
np.savez_compressed(OUT_SRC / 'object_poses.npz', world_from_object=Wn, world_from_rgb=d['world_from_rgb'], frames=d['frames'], names=d['names'])
chk = np.load(OUT_SRC / 'object_poses.npz'); assert np.array_equal(chk['world_from_rgb'], d['world_from_rgb']) and np.allclose(chk['world_from_object'], Wn)
shift = {}
for n in NAMES:
    e = np.linalg.norm(new_setup[n][:, :3, 3] - ours[n][:, :3, 3], axis=1) * 100; r = np.degrees(R.from_matrix(np.einsum('tji,tjk->tik', new_setup[n][:, :3, :3], ours[n][:, :3, :3])).magnitude())
    shift[n] = dict(position_cm_mean=float(e.mean()), position_cm_max=float(e.max()), rotation_deg_mean=float(r.mean()), rotation_deg_max=float(r.max()), frame0=[float(e[0]), float(r[0])], per_frame_cm=e.round(3).tolist(), per_frame_deg=r.round(2).tolist())
rec = dict(schema='ep19_compose_reference_correction_v1', episode=19,
    compose_tracks={n: dict(path=str(CMP / f'{n}_trajectory.npz'), sha256=sha(CMP / f'{n}_trajectory.npz'), mesh=str(CMP / f'{n}_mesh_metric.obj'), mesh_sha256=sha(CMP / f'{n}_mesh_metric.obj')) for n in NAMES},
    our_meshes={n: sha(SRC / f'assets/{n}_visual.obj') if (SRC / f'assets/{n}_visual.obj').exists() else None for n in NAMES},
    submitted_mesh_source=str(V21PRE), registration={n: dict(method='kit MP-SR _register_object_body (our packed mesh -> compose mesh, scorer seed 1000003*19+101*slot); our body pose = compose pose @ M', M=reg[n]['M'].tolist(), chamfer_mm=reg[n]['chamfer_mm']) for n in NAMES},
    world_transform=dict(clear_mm=A.clear_mm, keep_home=A.keep_home, X_setup_from_compose=X.tolist(), choice=A.x + ' frame-0 match; the scorer aligns on the frame-0 sweeper so a global rigid X does not change the compose score', setup_from_source=S.tolist()),
    sweeper_face='compose face = face of our un-relabelled source track (final_recon); the v14+ half-turn relabel G (recon_v15_20261022/episode_000019/change.json) must NOT be applied to this package',
    original_source=str(SRC), original_object_poses_sha256=sha(SRC / 'object_poses.npz'), corrected_object_poses_sha256=sha(OUT_SRC / 'object_poses.npz'),
    shift_vs_v21_reference_setup_world=shift, script=str(Path(__file__).resolve()), script_sha256=sha(__file__))
(OUT_SRC / 'compose_correction.json').write_text(json.dumps(rec, indent=1) + '\n')
# (c) scene: shipped frozen scene with the home keyframe object poses replaced
SCN = Path(man['physical_scene']); assert sha(SCN) == man['physical_scene_sha256']
txt = SCN.read_text(); m = mujoco.MjModel.from_xml_path(str(SCN)); dd = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, dd, 0)
q = dd.qpos.copy()
for n in NAMES:
    a = m.joint(n + '_free').qposadr[0]; T0 = new_setup[n][0].copy()
    if n in A.keep_home: print('keep shipped home keyframe', n); continue
    if A.clear_mm is not None:
        low = (np.asarray(packed[n][0]) @ T0[:3, :3].T + T0[:3, 3])[:, 2].min()
        if low < 0: T0[2, 3] += A.clear_mm / 1000 - low; print('keyframe lift', n, round((A.clear_mm / 1000 - low) * 100, 3), 'cm')
    q[a:a + 3] = T0[:3, 3]; q[a + 3:a + 7] = R.from_matrix(T0[:3, :3]).as_quat(scalar_first=True)
keys = re.findall(r'<key name="home" qpos="([^"]*)"', txt); assert len(keys) == 1
old = np.array(keys[0].split(), float); assert np.allclose(old, dd.qpos, atol=1e-5)
newq = ' '.join(repr(float(x)) if i >= m.joint('mini_sweeper_free').qposadr[0] else s for i, (x, s) in enumerate(zip(q, keys[0].split())))
OUT_SC.mkdir(parents=True); scn = OUT_SC / 'scene_physics_compose_home.xml'; scn.write_text(txt.replace(f'<key name="home" qpos="{keys[0]}"', f'<key name="home" qpos="{newq}"'))
m2 = mujoco.MjModel.from_xml_path(str(scn)); d2 = mujoco.MjData(m2); mujoco.mj_resetDataKeyframe(m2, d2, 0); mujoco.mj_forward(m2, d2)
diff = np.flatnonzero(np.abs(d2.qpos - dd.qpos) > 1e-12); assert diff.min() >= m.joint('mini_sweeper_free').qposadr[0]
contacts = []
for i in range(d2.ncon):
    c = d2.contact[i]; g1, g2 = m2.geom(c.geom1).name, m2.geom(c.geom2).name; contacts.append((g1, g2, float(c.dist)))
(OUT_SC / 'scene_change.json').write_text(json.dumps(dict(base_scene=str(SCN), base_scene_sha256=sha(SCN), new_scene=str(scn), new_scene_sha256=sha(scn), changed_qpos_indices=diff.tolist(),
    note='only the home keyframe object qpos (mini_sweeper_free, mini_dust_pan_free) changed: new frame-0 targets', initial_contacts=contacts), indent=1) + '\n')
# (b) episode dir
OUT_EP.mkdir(parents=True)
for f in EPDIR.iterdir():
    if f.is_file(): shutil.copy2(f, OUT_EP / f.name)
for f in ('tracking_reference.npz', 'reference_trajectory.npz'):
    z = dict(np.load(EPDIR / f, allow_pickle=False)); jn = list(z['joint_names'])
    for i, n in enumerate(NAMES):
        a = jn.index(n + '_free[0]'); P = new_setup[n]; z['qpos'][:, a:a + 3] = P[:, :3, 3]; z['qpos'][:, a + 3:a + 7] = R.from_matrix(P[:, :3, :3]).as_quat(scalar_first=True)
        if 'object_positions' in z:
            k = list(z['object_names']).index(n); z['object_positions'][:, k] = P[:, :3, 3]; z['object_quaternions_wxyz'][:, k] = R.from_matrix(P[:, :3, :3]).as_quat(scalar_first=True)
    np.savez_compressed(OUT_EP / f, **z)
m_ = dict(man); m_['source'] = str(OUT_SRC); m_['source_reference'] = str(OUT_SRC / 'reference.parquet'); m_['source_reference_sha256'] = sha(OUT_SRC / 'reference.parquet')
m_['source_object_poses_sha256'] = sha(OUT_SRC / 'object_poses.npz'); m_['physical_scene'] = str(scn); m_['physical_scene_sha256'] = sha(scn)
m_['source_object_targets_and_original_metric_anchors_unchanged'] = False
m_['compose_correction'] = dict(record=str(OUT_SRC / 'compose_correction.json'), record_sha256=sha(OUT_SRC / 'compose_correction.json'), script=str(Path(__file__).resolve()), script_sha256=sha(__file__),
    original_manifest=str(EPDIR / 'input_manifest.json'), original_manifest_sha256=sha(EPDIR / 'input_manifest.json'), original_source=str(SRC), original_source_object_poses_sha256=sha(SRC / 'object_poses.npz'),
    scene_change=str(OUT_SC / 'scene_change.json'), note='v21C hedge: object targets = coworker compose_eval poses (our meshes, scorer registration, one rigid X); reference.parquet (MANO/IK) unchanged copy')
(OUT_EP / 'input_manifest.json').write_text(json.dumps(m_, indent=2) + '\n')
print(json.dumps(dict(source=str(OUT_SRC), episode_dir=str(OUT_EP), scene=str(scn), object_poses_sha256=m_['source_object_poses_sha256'], shift={n: {k: v for k, v in shift[n].items() if not k.startswith('per_')} for n in NAMES}, contacts=contacts), indent=1))
