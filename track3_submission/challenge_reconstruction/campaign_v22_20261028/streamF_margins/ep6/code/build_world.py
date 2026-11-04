"""Stream A (phase 7, v22; copied from streamT_C356 with CMP = target_final_v5): build the target-referenced world for one legacy pot+spoon episode (3, 5, 6).

Reference = the coworker's compose_target poses (compose_eval_20261021), converted into OUR mesh body frames with the scorer's own
registration (MP-SR._register_object_body, same meshes/seed as score_vs_compose.py: T_ours = T_compose @ [I|anchor] @ inv([rot^T|ca]))
and placed in our physics (setup) frame by ONE rigid transform G per episode:
  G0  = the static-scene alignment that puts the compose pot frame-0 pose onto our pot frame-0 pose (the same anchor the official
        scorer uses: slot 0 = white_pot, frame 0),
  Gt  = a small table-plane correction: rotation about the frame-0 pot contact + height shift fitted so the compose RESTING contacts
        (pot bottom and spoon lowest point, median over the first 25 and last 10 frames) lie on our tabletop z = 0 (regularised
        least squares, prior 2 deg). G = Gt @ G0.
The world keeps our meshes, table, robot, controller; object targets are replaced (setup frame), the source copy carries the same
poses in the final_recon world (W_src = inv(setup_from_source) @ G @ T_compose_ourbody), and the scene home keyframe / object bodies
start at the new frame-0 poses.
Writes (new dirs only): streamT_C356/epE/world/source_compose/episode_X, streamT_C356/epE/world/episode_compose/episode_X,
streamT_C356/epE/world/compose_reference.json (lineage: compose sha256, registration, G0/Gt/G, contacts, per-frame shift vs v21 reference).
usage: build_world.py EP BASE_EPISODE_DIR"""
import sys, json, shutil, hashlib, re
from pathlib import Path
import numpy as np, trimesh, pyarrow.parquet as pq, pyarrow as pa
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, str(Path(__file__).resolve().parent)); import tlib as L
ep, base = int(sys.argv[1]), Path(sys.argv[2]).resolve()
name = f'episode_{ep:06d}'; T = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamF_margins') / f'ep{ep}' / 'world'
SRC = Path('/data1/SRL_V2D/challenge_reconstruction/final_recon') / name
OS, OE = T / 'source_compose' / name, T / 'episode_compose' / name
if OS.exists() or OE.exists(): raise SystemExit('exists')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
man = json.loads((base / 'input_manifest.json').read_text()); S = np.array(man['setup_from_source'])
assert Path(man['source']).resolve() == SRC.resolve() and sha(SRC / 'object_poses.npz') == man['source_object_poses_sha256']
z = dict(np.load(SRC / 'object_poses.npz', allow_pickle=False)); names = [str(x) for x in z['names']]; assert names == L.NAMES
W = z['world_from_object']; n = len(W)
Tc, X = L.compose_ourbody(ep); assert all(len(Tc[o]) == n for o in names), {o: len(Tc[o]) for o in names}
V = {o: np.asarray(trimesh.load(SRC / 'assets' / f'{o}.obj', process=False, force='mesh').vertices) for o in names}
mesh_sha = {o: sha(SRC / 'assets' / f'{o}.obj') for o in names}
assert mesh_sha['white_pot'].startswith('8cbd35431829') and mesh_sha['wooden_spoon'].startswith('c9ab88e33e5c'), mesh_sha
old = {o: S @ W[:, i] for i, o in enumerate(names)}            # v21 reference (setup frame)
G0 = old['white_pot'][0] @ np.linalg.inv(Tc['white_pot'][0])
REST = list(range(0, 25)), list(range(n - 10, n))
def contacts(G):
    out = []
    for win in REST:
        for o in names:
            zs, ps = [], []
            for f in win:
                Q = G @ Tc[o][f]; P = V[o] @ Q[:3, :3].T + Q[:3, 3]; k = P[:, 2] < P[:, 2].min() + (0.004 if o == 'white_pot' else 0.002)
                zs.append(P[:, 2].min()); ps.append(P[k].mean(0))
            out.append(dict(object=o, frames=[win[0], win[-1]], lowest_z_m=float(np.median(zs)), contact_xy_m=np.median(ps, 0)[:2].tolist()))
    return out
c0 = contacts(G0); pivot = np.r_[c0[0]['contact_xy_m'], 0.]
def Gt_of(x):
    M = np.eye(4); M[:3, :3] = R.from_rotvec([x[0], x[1], 0.]).as_matrix(); M[:3, 3] = pivot - M[:3, :3] @ pivot + [0, 0, x[2]]; return M
def resid(x):
    c = contacts(Gt_of(x) @ G0); return np.r_[[q['lowest_z_m'] / 0.003 for q in c], np.degrees(x[:2]) / 2.0]
fit = least_squares(resid, np.zeros(3), diff_step=1e-3); Gt = Gt_of(fit.x); G = Gt @ G0; c1 = contacts(G)
new = {o: G @ Tc[o] for o in names}
# per-frame shift of the new reference vs the v21 reference (setup frame, body origin / anchor-free)
shift = {}
for o in names:
    dp, dr = L.errs(new[o], old[o]); shift[o] = dict(mean_cm=float(dp.mean()), max_cm=float(dp.max()), mean_deg=float(dr.mean()), max_deg=float(dr.max()),
        per_second_cm=[round(float(dp[s:s + 20].mean()), 2) for s in range(0, n, 20)], per_second_deg=[round(float(dr[s:s + 20].mean()), 1) for s in range(0, n, 20)])
# ---- source copy (final_recon world) ----
Wn = np.stack([np.linalg.inv(S) @ new[o] for o in names], 1)
shutil.copytree(SRC, OS, symlinks=True); z['world_from_object'] = Wn; (OS / 'object_poses.npz').unlink(); np.savez_compressed(OS / 'object_poses.npz', **z)
for i, o in enumerate(names):
    tr = dict(np.load(SRC / f'{o}_trajectory.npz', allow_pickle=False)); tr['T_world_obj'] = Wn[:, i].copy(); tr['T_rgb_obj'] = np.linalg.inv(z['world_from_rgb']) @ Wn[:, i]
    (OS / f'{o}_trajectory.npz').unlink(); np.savez_compressed(OS / f'{o}_trajectory.npz', **tr)
tab = pq.read_table(SRC / 'reference.parquet'); rows = tab.to_pylist(); assert len(rows) == 1
rows[0]['object_body_position'] = Wn[:, :, :3, 3].tolist()
rows[0]['object_body_wxyz'] = R.from_matrix(Wn[:, :, :3, :3].reshape(-1, 3, 3)).as_quat(scalar_first=True).reshape(n, len(names), 4).tolist()
(OS / 'reference.parquet').unlink(); pq.write_table(pa.Table.from_pylist(rows, schema=tab.schema), OS / 'reference.parquet')
# ---- episode dir copy (setup frame) ----
shutil.copytree(base, OE, symlinks=True)
qn = {o: R.from_matrix(new[o][:, :3, :3]).as_quat(scalar_first=True) for o in names}
def fix_q(q, jn):
    q = q.copy()
    for o in names:
        c = jn.index(f'{o}_free'); 
        q[:, c:c + 3] = new[o][:, :3, 3]; s = np.sign((qn[o] * q[:, c + 3:c + 7]).sum(1, keepdims=True)); s[s == 0] = 1; q[:, c + 3:c + 7] = qn[o] * s
    return q
t = dict(np.load(base / 'tracking_reference.npz', allow_pickle=False)); jn = [str(x).replace('[0]', '') for x in t['joint_names']]
# joint names are per qpos column ('wooden_spoon_free[0]'..[6]) or per joint
jcols = [str(x) for x in t['joint_names']]
def col(o): return jcols.index(f'{o}_free[0]') if f'{o}_free[0]' in jcols else jcols.index(f'{o}_free')
def fix(q):
    q = q.copy()
    for o in names:
        c = col(o); q[:, c:c + 3] = new[o][:, :3, 3]; s = np.sign((qn[o] * q[:, c + 3:c + 7]).sum(1, keepdims=True)); s[s == 0] = 1; q[:, c + 3:c + 7] = qn[o] * s
    return q
t['qpos'] = fix(t['qpos']); (OE / 'tracking_reference.npz').unlink(); np.savez_compressed(OE / 'tracking_reference.npz', **t)
rt = dict(np.load(base / 'reference_trajectory.npz', allow_pickle=False)); assert [str(x) for x in rt['joint_names']] == jcols
rt['qpos'] = fix(rt['qpos']); rt['object_positions'] = np.stack([new[o][:, :3, 3] for o in names], 1); rt['object_quaternions_wxyz'] = np.stack([qn[o] for o in names], 1)
(OE / 'reference_trajectory.npz').unlink(); np.savez_compressed(OE / 'reference_trajectory.npz', **rt)
q0 = t['qpos'][0]
def fix_scene(src_xml, dst_xml):
    """object bodies + home keyframe start at the new frame-0 poses (everything else byte-identical)."""
    s = Path(src_xml).read_text()
    for o in names:
        p = new[o][0, :3, 3]; qq = qn[o][0]
        s, k = re.subn(rf'(<body name="{o}" )pos="[^"]*" quat="[^"]*"', lambda m: m.group(1) + f'pos="{p[0]:.9g} {p[1]:.9g} {p[2]:.9g}" quat="{qq[0]:.9g} {qq[1]:.9g} {qq[2]:.9g} {qq[3]:.9g}"', s); assert k == 1, (o, k)
    import mujoco
    mm = mujoco.MjModel.from_xml_string(s) if False else None
    kq = re.search(r'<key name="home" qpos="([^"]*)"', s); vals = np.array(kq.group(1).split(), float)
    for o in names:
        c = col(o); vals[c:c + 3] = new[o][0, :3, 3]; vals[c + 3:c + 7] = qn[o][0]
    s = s[:kq.start(1)] + ' '.join(f'{v:.9g}' for v in vals) + s[kq.end(1):]
    Path(dst_xml).write_text(s)
(OE / 'stage2/scene.xml').unlink(); fix_scene(base / 'stage2/scene.xml', OE / 'stage2/scene.xml')
rec = dict(episode=ep, schema='streamA_v22_target_final_v5_reference_v1', compose_dir=f'{L.CMP}/{name}',
    compose_sha256={o: sha(f'{L.CMP}/{name}/{o}_trajectory.npz') for o in names} | {o + '_mesh': sha(f'{L.CMP}/{name}/{o}_mesh_metric.obj') for o in names},
    our_mesh_sha256=mesh_sha, registration_T_ourbody_from_composebody={o: X[o].tolist() for o in names},
    registration_note='scorer registration (MP-SR._register_object_body, seed 1000003*ep+101*slot, our v21 submitted mesh vs compose mesh); T_ours = T_compose @ X',
    G0_setup_from_compose=G0.tolist(), Gt_table_correction=Gt.tolist(), G_setup_from_compose=G.tolist(),
    Gt_tilt_deg=float(np.degrees(np.linalg.norm(fit.x[:2]))), Gt_dz_cm=float(fit.x[2] * 100), Gt_rotvec_deg=np.degrees(fit.x[:2]).tolist(), pivot_xy_m=pivot[:2].tolist(),
    resting_contacts_before=c0, resting_contacts_after=c1, shift_vs_v21_reference_setup=shift,
    original_source=str(SRC), original_object_poses_sha256=sha(SRC / 'object_poses.npz'), corrected_object_poses_sha256=sha(OS / 'object_poses.npz'),
    base_episode_dir=str(base), base_manifest_sha256=sha(base / 'input_manifest.json'), script=str(Path(__file__).resolve()), script_sha256=sha(__file__),
    tlib_sha256=sha(Path(__file__).resolve().parent / 'tlib.py'))
(T / 'compose_reference.json').write_text(json.dumps(rec, indent=1)); shutil.copy2(T / 'compose_reference.json', OS / 'compose_reference.json')
man['source'] = str(OS); man['source_reference'] = str(OS / 'reference.parquet'); man['source_reference_sha256'] = sha(OS / 'reference.parquet')
man['source_object_poses_sha256'] = sha(OS / 'object_poses.npz')
man['corrected_track_note'] = 'v22: object targets = our reconstruction selected20_final_v5 (target_final_v5) poses in our mesh frames, one rigid transform G (pot frame-0 anchor + table-plane correction); robot/scene/meshes unchanged; see compose_reference.json'
man['compose_correction'] = dict(record=str(OS / 'compose_reference.json'), record_sha256=sha(OS / 'compose_reference.json'), original_source=str(SRC),
    original_source_object_poses_sha256=sha(SRC / 'object_poses.npz'), original_manifest=str(base / 'input_manifest.json'), original_manifest_sha256=sha(base / 'input_manifest.json'))
(OE / 'input_manifest.json').write_text(json.dumps(man, indent=2) + '\n')
print(json.dumps(dict(G0_vs_G_tilt_deg=rec['Gt_tilt_deg'], dz_cm=rec['Gt_dz_cm'], contacts_before=[round(c['lowest_z_m'] * 100, 2) for c in c0], contacts_after=[round(c['lowest_z_m'] * 100, 2) for c in c1],
    shift={o: (round(v['mean_cm'], 2), round(v['mean_deg'], 1), round(v['max_cm'], 1)) for o, v in shift.items()})))
