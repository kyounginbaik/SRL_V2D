"""Stream G (phase 7, v22): build the target_final_v5-referenced world for episode 1 (white_pot + lid) or 20 (blue_cup + water_pitcher).
Generalised from streamT_C356/code/build_world.py (same method):
  reference = target_final_v5 poses converted into OUR mesh body frames with the scorer's own registration (dlib.registration),
  placed in our physics (setup) frame by ONE rigid transform G = Gt @ G0:
  G0 = puts the target ANCHOR object's frame-0 pose onto the physics frame-0 pose of that object (taken from the teacher's
       execution initial_qpos = the scene home the rollout really starts from; ep1 anchor white_pot (= scorer anchor slot 0),
       ep20 anchor water_pitcher (the scorer takes the alignment rotation from the pitcher, the only non-symmetric object)),
  Gt = table-plane correction (rotation about the anchor's frame-0 contact + height) fitted so the target RESTING contacts
       (per-episode windows below) lie on our tabletop z = 0 (regularised LS, prior 2 deg).
The rollout's anchor starts at G @ target anchor f0, so the scorer's anchor alignment undoes G.
Writes (new dirs only): streamG_ep10/epE/world/{source_target,episode_target}/episode_X and world/target_reference.json.
usage: build_world_g.py EP BASE_EPISODE_DIR TEACHER_DIR [OUT_WORLD_DIR]"""
import sys, json, shutil, hashlib, re
from pathlib import Path
import numpy as np, trimesh, pyarrow.parquet as pq, pyarrow as pa
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG_ep10/code'); import glib as L   # stream G helpers (unchanged)
ep, base, teacher = int(sys.argv[1]), Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve()
names = L.NAMES[ep]; name = f'episode_{ep:06d}'
ANCHOR = {10: 'blue_cup'}[ep]
D = Path(sys.argv[4]).resolve() if len(sys.argv) > 4 else Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG_ep10') / 'world'
OS, OE = D / 'source_target' / name, D / 'episode_target' / name
if OS.exists() or OE.exists(): raise SystemExit('exists')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
man = json.loads((base / 'input_manifest.json').read_text()); S = np.array(json.loads(man['setup_from_source']) if isinstance(man['setup_from_source'], str) else man['setup_from_source'])
SRC = Path(man['source']).resolve(); assert sha(SRC / 'object_poses.npz') == man['source_object_poses_sha256']
z = dict(np.load(SRC / 'object_poses.npz', allow_pickle=False)); snames = [str(x) for x in z['names']]; assert snames == names, snames
W = z['world_from_object']; n = len(W)
Tc, X = L.target_ourbody(ep); assert all(len(Tc[o]) == n for o in names), {o: len(Tc[o]) for o in names}
def asset(o):
    p = SRC / 'assets' / f'{o}.obj'; return p if p.exists() else SRC / 'assets' / f'{o}_visual.obj'
V = {o: np.asarray(trimesh.load(asset(o), process=False, force='mesh').vertices) for o in names}
mesh_sha = {o: sha(asset(o)) for o in names}
EXPECT = {'blue_cup': '57bb269a06a4', 'beige_cup': '2c041e69fbd3'}
assert all(mesh_sha[o].startswith(EXPECT[o]) for o in names), mesh_sha
old = {o: S @ W[:, i] for i, o in enumerate(names)}            # v21 reference (setup frame)
# physics frame-0 pose of each object = teacher execution initial_qpos
tres = json.loads((teacher / 'result.json').read_text()); trj = np.load(tres['execution']['trajectory'])
import mujoco
m = mujoco.MjModel.from_xml_path(tres['execution']['scene']); q0 = trj['initial_qpos'] if 'initial_qpos' in trj else trj['qpos'][0]
def body0(o):
    c = int(m.joint(o + '_free').qposadr[0]); T = np.eye(4); T[:3, 3] = q0[c:c + 3]; T[:3, :3] = R.from_quat(q0[c + 3:c + 7], scalar_first=True).as_matrix(); return T
P0 = {o: body0(o) for o in names}
# ep10: both cups are axisymmetric (scorer order 360), so the scorer aligns frame 0 by the blue_cup axis + the frame-0 direction
# blue -> beige (MP-SR._align_registered_scene, all-symmetric branch) and the blue anchor point.  G0 uses the same rule: it maps
# the target frame-0 layout onto our physics frame-0 layout (blue axis, blue->beige direction, blue anchor point).
sys.path.insert(0, '/data1/SRL_V2D/challenge_submissions_SRL/analysis_20261020/scripts'); import public_kit_score as K0
_ab = X['blue_cup'][:3, :3].T @ K0.symmetry_axis('blue_cup', L.ref_mesh(ep, 'blue_cup')[0])[0]; _ab /= np.linalg.norm(_ab)
_pt = lambda T, o: T[:3, :3] @ L.CA[(ep, o)] + T[:3, 3]
_Rp = L.M._axis_basis(P0['blue_cup'][:3, :3] @ _ab, _pt(P0['beige_cup'], 'beige_cup') - _pt(P0['blue_cup'], 'blue_cup'))
_Rt = L.M._axis_basis(Tc['blue_cup'][0][:3, :3] @ _ab, _pt(Tc['beige_cup'][0], 'beige_cup') - _pt(Tc['blue_cup'][0], 'blue_cup'))
G0 = np.eye(4); G0[:3, :3] = _Rp @ _Rt.T; G0[:3, 3] = _pt(P0['blue_cup'], 'blue_cup') - G0[:3, :3] @ _pt(Tc['blue_cup'][0], 'blue_cup')
REST = {10: [('blue_cup', range(0, 25)), ('beige_cup', range(0, 25)), ('blue_cup', range(n - 20, n))]}[ep]
def contacts(G):
    out = []
    for o, win in REST:
        zs, ps = [], []
        for f in win:
            Q = G @ Tc[o][f]; P = V[o] @ Q[:3, :3].T + Q[:3, 3]; k = P[:, 2] < P[:, 2].min() + 0.003
            zs.append(P[:, 2].min()); ps.append(P[k].mean(0))
        out.append(dict(object=o, frames=[win[0], win[-1]], lowest_z_m=float(np.median(zs)), contact_xy_m=np.median(ps, 0)[:2].tolist()))
    return out
c0 = contacts(G0); pivot = np.r_[c0[0]['contact_xy_m'], 0.]
# N8: the target puts both cups 6-8 deg off the physics table normal at rest (f0-24), so in the stream G world they rock upright at
# the start (1.5 rad/s, ~1 cm lateral) and keep a tilt error.  UP = minimal rotation taking the mean target cup axis at rest onto the
# table normal (+-z); Gt = height fit only, applied after UP (pivot = blue contact).
_ax = {o: X[o][:3, :3].T @ K0.symmetry_axis(o, L.ref_mesh(ep, o)[0])[0] for o in names}
_w = []
for o in names:
    for f in range(0, 25):
        a = (G0 @ Tc[o][f])[:3, :3] @ _ax[o]; _w.append(a / np.linalg.norm(a) * np.sign(a[2]))
_w = np.mean(_w, 0); _w /= np.linalg.norm(_w); _c = np.cross(_w, [0, 0, 1.]); UPrv = _c / np.linalg.norm(_c) * np.arccos(np.clip(_w[2], -1, 1))
UP_deg = float(np.degrees(np.linalg.norm(UPrv)))
def Gt_of(x):
    M = np.eye(4); M[:3, :3] = R.from_rotvec(UPrv).as_matrix(); M[:3, 3] = pivot - M[:3, :3] @ pivot + [0, 0, x[2]]; return M
# physical rest heights of our visual meshes in the teacher rollout (same windows) are the targets (collision hulls keep the visual
# mesh a few mm above z = 0)
nm_t = [str(x) for x in trj['object_names']]
def phys_lowz(o, win):
    i = nm_t.index(o); out = []
    for f in win:
        Q = np.eye(4); Q[:3, 3] = trj['object_positions'][f, i]; Q[:3, :3] = R.from_quat(trj['object_quaternions_wxyz'][f, i][[1, 2, 3, 0]]).as_matrix()
        out.append((V[o] @ Q[:3, :3].T + Q[:3, 3])[:, 2].min())
    return float(np.median(out))
ZT = np.array([phys_lowz(o, win) for o, win in REST])
def resid(x):
    c = contacts(Gt_of(x) @ G0); return np.r_[[(q['lowest_z_m'] - zt) / 0.003 for q, zt in zip(c, ZT)]]
fit = least_squares(resid, np.zeros(3), diff_step=1e-3); Gt = Gt_of(fit.x); G = Gt @ G0; c1 = contacts(G)
new = {o: G @ Tc[o] for o in names}
# axisymmetric objects (scorer order 360: blue_cup): the scorer ignores spin about the symmetry axis, and the target spin is not
# observable (it differs from ours by up to 150 deg).  Keep OUR reference spin: rotation = minimal swing (old axis -> target axis)
# applied to the v21 reference rotation; position chosen so the scorer's anchor point (ca, our body coords) is exactly the target's.
sym_note = {}
sys.path.insert(0, '/data1/SRL_V2D/challenge_submissions_SRL/analysis_20261020/scripts'); import public_kit_score as K
for o in names:
    if K.SYMMETRY.get(o) != 360: continue
    vt, _ = L.ref_mesh(ep, o); ax_t, _ = K.symmetry_axis(o, vt); a = X[o][:3, :3].T @ ax_t; a /= np.linalg.norm(a); ca = L.CA[(ep, o)]
    pw = np.einsum('nij,j->ni', new[o][:, :3, :3], ca) + new[o][:, :3, 3]
    Ro = old[o][:, :3, :3]; ao = Ro @ a; an = new[o][:, :3, :3] @ a
    cr = np.cross(ao, an); sn = np.linalg.norm(cr, axis=1, keepdims=True); cs = (ao * an).sum(1)
    rv = cr / np.maximum(sn, 1e-12) * np.arctan2(sn[:, 0], cs)[:, None]
    Rn = R.from_rotvec(rv).as_matrix() @ Ro
    T = np.tile(np.eye(4), (n, 1, 1)); T[:, :3, :3] = Rn; T[:, :3, 3] = pw - np.einsum('nij,j->ni', Rn, ca)
    sym_note[o] = dict(axis_ourbody=a.tolist(), anchor_ourbody=ca.tolist(), swing_deg_mean=float(np.degrees(np.linalg.norm(rv, axis=1)).mean()),
                       target_spin_discarded_deg_mean=float(L.errs(new[o], T)[1].mean()))
    new[o] = T
shift = {}
for o in names:
    dp, dr = L.errs(new[o], old[o]); shift[o] = dict(mean_cm=float(dp.mean()), max_cm=float(dp.max()), mean_deg=float(dr.mean()), max_deg=float(dr.max()),
        per_second_cm=[round(float(dp[s:s + 20].mean()), 2) for s in range(0, n, 20)], per_second_deg=[round(float(dr[s:s + 20].mean()), 1) for s in range(0, n, 20)])
f0_vs_physics = {o: dict(zip(('cm', 'deg'), [float(v[0]) for v in L.errs(new[o][:1], P0[o][None])])) for o in names}
# ---- source copy (source world) ----
Wn = np.stack([np.linalg.inv(S) @ new[o] for o in names], 1)
shutil.copytree(SRC, OS, symlinks=True); z['world_from_object'] = Wn; (OS / 'object_poses.npz').unlink(); np.savez_compressed(OS / 'object_poses.npz', **z)
for i, o in enumerate(names):
    if not (SRC / f'{o}_trajectory.npz').exists(): continue
    tr = dict(np.load(SRC / f'{o}_trajectory.npz', allow_pickle=False)); tr['T_world_obj'] = Wn[:, i].copy()
    if 'T_rgb_obj' in tr and 'world_from_rgb' in z: tr['T_rgb_obj'] = np.linalg.inv(z['world_from_rgb']) @ Wn[:, i]
    (OS / f'{o}_trajectory.npz').unlink(); np.savez_compressed(OS / f'{o}_trajectory.npz', **tr)
tab = pq.read_table(SRC / 'reference.parquet'); rows = tab.to_pylist(); assert len(rows) == 1
bn = list(rows[0]['object_body_names']); assert bn == names, bn
rows[0]['object_body_position'] = Wn[:, :, :3, 3].tolist()
rows[0]['object_body_wxyz'] = R.from_matrix(Wn[:, :, :3, :3].reshape(-1, 3, 3)).as_quat(scalar_first=True).reshape(n, len(names), 4).tolist()
(OS / 'reference.parquet').unlink(); pq.write_table(pa.Table.from_pylist(rows, schema=tab.schema), OS / 'reference.parquet')
# ---- episode dir copy (setup frame) ----
shutil.copytree(base, OE, symlinks=True)
qn = {o: R.from_matrix(new[o][:, :3, :3]).as_quat(scalar_first=True) for o in names}
t = dict(np.load(base / 'tracking_reference.npz', allow_pickle=False)); jcols = [str(x) for x in t['joint_names']]
def col(o): return jcols.index(f'{o}_free[0]') if f'{o}_free[0]' in jcols else jcols.index(f'{o}_free')
def fix(q):
    q = q.copy()
    for o in names:
        c = col(o); q[:, c:c + 3] = new[o][:, :3, 3]; s = np.sign((qn[o] * q[:, c + 3:c + 7]).sum(1, keepdims=True)); s[s == 0] = 1; q[:, c + 3:c + 7] = qn[o] * s
    return q
t['qpos'] = fix(t['qpos']); (OE / 'tracking_reference.npz').unlink(); np.savez_compressed(OE / 'tracking_reference.npz', **t)
if (base / 'reference_trajectory.npz').exists():
    rt = dict(np.load(base / 'reference_trajectory.npz', allow_pickle=False)); assert [str(x) for x in rt['joint_names']] == jcols
    rt['qpos'] = fix(rt['qpos']); rt['object_positions'] = np.stack([new[o][:, :3, 3] for o in names], 1); rt['object_quaternions_wxyz'] = np.stack([qn[o] for o in names], 1)
    (OE / 'reference_trajectory.npz').unlink(); np.savez_compressed(OE / 'reference_trajectory.npz', **rt)
def fix_scene(src_xml, dst_xml):
    s = Path(src_xml).read_text()
    for o in names:
        p = new[o][0, :3, 3]; qq = qn[o][0]
        s, k = re.subn(rf'(<body name="{o}" )pos="[^"]*" quat="[^"]*"', lambda mm: mm.group(1) + f'pos="{p[0]:.9g} {p[1]:.9g} {p[2]:.9g}" quat="{qq[0]:.9g} {qq[1]:.9g} {qq[2]:.9g} {qq[3]:.9g}"', s)
        if k != 1: print('fix_scene: no object body', o, 'in', src_xml, '-> file copied unchanged'); return False
    kq = re.search(r'<key name="home" qpos="([^"]*)"', s)
    if kq:
        vals = np.array(kq.group(1).split(), float); mm = mujoco.MjModel.from_xml_path(str(src_xml))
        for o in names:
            c = int(mm.joint(o + '_free').qposadr[0]); vals[c:c + 3] = new[o][0, :3, 3]; vals[c + 3:c + 7] = qn[o][0]
        s = s[:kq.start(1)] + ' '.join(f'{v:.9g}' for v in vals) + s[kq.end(1):]
    Path(dst_xml).unlink(); Path(dst_xml).write_text(s); return True
for rel in ('stage2/scene.xml', 'setup/scene.xml'):
    if (base / rel).exists() and not (base / rel).is_symlink():
        print(rel, 'rewritten' if fix_scene(base / rel, OE / rel) else 'unchanged')
rec = dict(episode=ep, schema='streamG_v22_target_final_v5_reference_v1', target_dir=f'{L.CMP}/{name}', anchor=ANCHOR,
    target_sha256={o: sha(f'{L.CMP}/{name}/{o}_trajectory.npz') for o in names} | {o + '_mesh': sha(f'{L.CMP}/{name}/{o}_mesh_metric.obj') for o in names},
    our_mesh_sha256=mesh_sha, registration_T_ourbody_from_targetbody={o: X[o].tolist() for o in names},
    registration_note='scorer registration (MP-SR._register_object_body, seed 1000003*ep+101*slot, our v21 submitted mesh vs target mesh); T_ours = T_target @ X',
    physics_frame0_from=str(teacher), axisymmetric_spin_kept_from_v21_reference=sym_note, G0_setup_from_target=G0.tolist(), Gt_table_correction=Gt.tolist(), G_setup_from_target=G.tolist(),
    Gt_tilt_deg=UP_deg, UP_rotvec_deg=np.degrees(UPrv).tolist(), Gt_dz_cm=float(fit.x[2] * 100), Gt_rotvec_deg=np.degrees(fit.x[:2]).tolist(), pivot_xy_m=pivot[:2].tolist(),
    resting_contacts_before=c0, resting_contacts_after=c1, physical_rest_lowest_z_m=ZT.tolist(), shift_vs_v21_reference_setup=shift, frame0_vs_physics_initial=f0_vs_physics,
    original_source=str(SRC), original_object_poses_sha256=sha(SRC / 'object_poses.npz'), corrected_object_poses_sha256=sha(OS / 'object_poses.npz'),
    base_episode_dir=str(base), base_manifest_sha256=sha(base / 'input_manifest.json'), script=str(Path(__file__).resolve()), script_sha256=sha(__file__),
    glib_sha256=sha('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG_ep10/code/glib.py'))
(D / 'target_reference.json').write_text(json.dumps(rec, indent=1)); shutil.copy2(D / 'target_reference.json', OS / 'target_reference.json')
man['source'] = str(OS); man['source_reference'] = str(OS / 'reference.parquet'); man['source_reference_sha256'] = sha(OS / 'reference.parquet')
man['source_object_poses_sha256'] = sha(OS / 'object_poses.npz')
man['corrected_track_note'] = 'v22: object targets = our reconstruction selected20_final_v5 (target_final_v5) poses in our mesh frames, one rigid transform G (anchor frame-0 + table-plane correction); robot/scene/meshes unchanged; see target_reference.json'
man['target_correction'] = dict(record=str(OS / 'target_reference.json'), record_sha256=sha(OS / 'target_reference.json'), original_source=str(SRC),
    original_source_object_poses_sha256=sha(SRC / 'object_poses.npz'), original_manifest=str(base / 'input_manifest.json'), original_manifest_sha256=sha(base / 'input_manifest.json'))
(OE / 'input_manifest.json').write_text(json.dumps(man, indent=2) + '\n')
print(json.dumps(dict(Gt_tilt_deg=rec['Gt_tilt_deg'], dz_cm=rec['Gt_dz_cm'], contacts_before=[round(c['lowest_z_m'] * 100, 2) for c in c0], contacts_after=[round(c['lowest_z_m'] * 100, 2) for c in c1], physical_rest=[round(v * 100, 2) for v in ZT],
    f0_vs_physics={o: (round(v['cm'], 2), round(v['deg'], 1)) for o, v in f0_vs_physics.items()},
    shift={o: (round(v['mean_cm'], 2), round(v['mean_deg'], 1), round(v['max_cm'], 1)) for o, v in shift.items()})))
