"""[T33 copy of streamB build_world_b.py: reference = $E31_TGT (e.g. R33_HGA_r6) via blib.TGT override; --src DIR = previous physics layout W0 + source copied (default final_recon as before); everything else unchanged] Stream B (v22) world builder for eps 32/33 vs target_final_v5 (copy of streamR compose_retarget32_v3.py, generalized: EP param, the
reference = campaign_v22_20261028/target_final_v5 (our recon, canonical world), body registration = the scorer's own MP-SR
registration of the SHIPPED meshes (--meshes DIR with blue_cup.obj / wooden_spoon.obj, budgeted exactly as make_sub_any.py does), object
qpos addresses read from the model). G = Gt @ G0: G0 = frame-0 Kabsch of both objects onto our previous physics layout (W0 = SRC object_poses),
Gt = minimal tilt about the frame-0 cup base + vertical shift putting the shipped cup bottom (57bb bottom at -SHORT) and the lowest
spoon point on the table z=0. Cup yaw about its symmetry axis fixed to ours (scoring-invariant).
v3 docstring follows.
(v3 = v2 with a two-point table fit: instead of standing the frame-0 cup upright, the minimal tilt about the frame-0
cup base (axis horizontal, perpendicular to the cup->spoon direction) plus a vertical shift puts BOTH the shipped cup bottom (57bb bottom at -SHORT)
and the lowest spoon point on the table z=0). v2 docstring: the compose frame-0 layout differs from ours (best rigid fit leaves the spoon 14 deg
rotated relative to the cup), and the official scorer aligns on the frame-0 spoon rotation, so the physics scene must START from the
compose frame-0 poses. Steps:
 1. Kit-scorer body registration (as v1) + cup yaw about its symmetry axis (local z) fixed to our frame-0 cup yaw (scoring-invariant).
 2. World transform X = v1 Kabsch (frame-0 mesh points, our layout) then, in the setup frame, a tilt about the frame-0 cup base making the
    compose frame-0 cup axis vertical and a vertical shift putting the SHIPPED cup (3c3c = 57bb shortened 1.78 cm at the bottom) on the table
    (57bb bottom at -SHORT). The compose camera/scale is adopted as given.
 3. Corrected source copy (object_poses both objects, compose_retarget.json) + episode dir (manifest -> source; tracking_reference and
    reference_trajectory object columns for ALL frames incl. frame 0) + a new physics scene = copy of BASE_SCENE with the two object bodies'
    pos/quat and keyframe object qpos set to the new frame-0 poses (robot, meshes, contacts, everything else unchanged).
usage: build_world_b.py EP OUT_DIR BASE_SCENE_XML --base-epd DIR --meshes DIR [--short 0.0178]"""
import os, sys, json, shutil, hashlib, argparse, re, numpy as np, trimesh, pandas as pd
from pathlib import Path
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, '/data1/SRL_V2D/challenge_submissions_SRL/analysis_20261020/scripts'); import public_kit_score as K
sys.path.insert(0, str(Path(__file__).resolve().parent)); import blib as BL
BL.TGT = Path(os.environ["E31_TGT"])   # T33: reference = $E31_TGT (our recon target) instead of target_final_v5
ap = argparse.ArgumentParser(); ap.add_argument('ep', type=int); ap.add_argument('out'); ap.add_argument('scene'); ap.add_argument('--short', type=float, default=0.0178)
ap.add_argument('--base-epd', required=True); ap.add_argument('--src'); ap.add_argument('--meshes', required=True); a = ap.parse_args(); EP = a.ep; EN = f'episode_{EP:06d}'
out = Path(a.out).resolve(); OS, OE = out / f'source_corrected/{EN}', out / f'episode_corrected/{EN}'
if OS.exists() or OE.exists(): raise SystemExit('exists')
SRC = Path(a.src) if a.src else Path(f'/data1/SRL_V2D/challenge_reconstruction/final_recon/{EN}'); CMP = BL.TGT / EN   # T33: --src = previous physics layout + source to copy
EPD = Path(a.base_epd); V21 = '/data1/SRL_V2D/challenge_submissions/track3_v21_20261024/all_20_episodes_post/track3_submission.parquet'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
M = K.load_metric('MP-SR'); names = ['blue_cup', 'wooden_spoon']
G = BL.G_ourbody(EP, BL.register(EP, {n: Path(a.meshes) / f'{n}.obj' for n in names}))
z = dict(np.load(SRC / 'object_poses.npz', allow_pickle=False)); zn = [str(x) for x in z['names']]; W0 = z['world_from_object'].copy(); N = len(W0)
To = {n: np.load(CMP / f'{n}_trajectory.npz')['T_world_obj'].astype(np.float64) @ np.linalg.inv(G[n]) for n in names}
pts = {n: trimesh.load(SRC / f'assets/{n}.obj', force='mesh').sample(4000, seed=0) if 'seed' in trimesh.Trimesh.sample.__code__.co_varnames else trimesh.load(SRC / f'assets/{n}.obj', force='mesh').sample(4000) for n in names}
def kabsch(A, B):
    ca_, cb_ = A.mean(0), B.mean(0); U, S_, Vt = np.linalg.svd((A - ca_).T @ (B - cb_)); Rx = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
    X = np.eye(4); X[:3, :3] = Rx; X[:3, 3] = cb_ - Rx @ ca_; return X
X1 = kabsch(np.vstack([pts[n] @ To[n][0, :3, :3].T + To[n][0, :3, 3] for n in names]), np.vstack([pts[n] @ W0[0, zn.index(n), :3, :3].T + W0[0, zn.index(n), :3, 3] for n in names]))
# cup yaw about its local z axis to our frame-0 yaw (symmetric object; scoring-invariant)
ci, si = zn.index('blue_cup'), zn.index('wooden_spoon')
D = (X1 @ To['blue_cup'][0])[:3, :3].T @ W0[0, ci, :3, :3]; yaw = np.arctan2(D[1, 0] - D[0, 1], D[0, 0] + D[1, 1])
Ry = np.eye(4); Ry[:3, :3] = R.from_euler('z', yaw).as_matrix(); To['blue_cup'] = To['blue_cup'] @ Ry; G['blue_cup'] = np.linalg.inv(Ry) @ G['blue_cup']
Sm = np.asarray(json.loads((EPD / 'input_manifest.json').read_text())['setup_from_source']); Sinv = np.linalg.inv(Sm)
Vc = np.asarray(trimesh.load(SRC / 'assets/blue_cup.obj', force='mesh').vertices); Vs = np.asarray(trimesh.load(SRC / 'assets/wooden_spoon.obj', force='mesh').vertices)
C0 = Sm @ X1 @ To['blue_cup'][0]; S0 = Sm @ X1 @ To['wooden_spoon'][0]
cw = Vc @ C0[:3, :3].T + C0[:3, 3]; sw = Vs @ S0[:3, :3].T + S0[:3, 3]
pivot = cw[cw[:, 2] < cw[:, 2].min() + 0.003].mean(0); ps = sw[np.argmin(sw[:, 2])]
hd = ps[:2] - pivot[:2]; L = np.linalg.norm(hd); u = np.array([-hd[1], hd[0], 0.]) / L     # tilt axis (horizontal, perpendicular to cup->spoon)
need = (ps[2] - pivot[2]) - (0.0 - (-a.short))          # spoon lowest point must end a.short above the 57bb cup bottom
ang = np.arctan2(need, L); Rt = R.from_rotvec(u * ang).as_matrix()
T1 = np.eye(4); T1[:3, :3] = Rt; T1[:3, 3] = pivot - Rt @ pivot
cw2 = Vc @ (T1 @ C0)[:3, :3].T + (T1 @ C0)[:3, 3]; dz = -a.short - cw2[:, 2].min(); T2 = np.eye(4); T2[2, 3] = dz
X = Sinv @ T2 @ T1 @ Sm @ X1
W = W0.copy()
for n in names: W[:, zn.index(n)] = X @ To[n]
anc = {n: trimesh.load(SRC / f'assets/{n}.obj', force='mesh', process=False).centroid for n in names}
def bottoms(n, V, k): T = Sm @ W[k, zn.index(n)]; return float((V @ T[:3, :3].T + T[:3, 3])[:, 2].min() * 100)
chk = dict(tilt_applied_deg=float(np.degrees(ang)), lift_cm=float(dz * 100), cup_yaw_fix_deg=float(np.degrees(yaw)),
           cup57bb_bottom_cm={k: bottoms('blue_cup', Vc, k) for k in (0, 100, 200, 258)}, spoon_bottom_cm={k: bottoms('wooden_spoon', Vs, k) for k in (0, 50, 200, 258)},
           cup_axis_tilt_f0_end_deg=[float(np.degrees(np.arccos(abs((Sm @ W[k, ci])[2, 2])))) for k in (0, 258)])
shift = {n: (np.linalg.norm((W[:, zn.index(n), :3, :3] @ anc[n] + W[:, zn.index(n), :3, 3]) - (W0[:, zn.index(n), :3, :3] @ anc[n] + W0[:, zn.index(n), :3, 3]), axis=1) * 100) for n in names}
rot = {n: np.degrees([R.from_matrix(x.T @ y).magnitude() for x, y in zip(W[:, zn.index(n), :3, :3], W0[:, zn.index(n), :3, :3])]) for n in names}
print(json.dumps(chk)); print('shift vs v21 reference mean cm/deg', {n: (round(float(shift[n].mean()), 2), round(float(rot[n].mean()), 1)) for n in names}, 'frame0', {n: (round(float(shift[n][0]), 2), round(float(rot[n][0]), 1)) for n in names})
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
# scene
xml = Path(a.scene).read_text(); newq = {}
for n in names:
    TS = Sm @ W[0, zn.index(n)]; q = R.from_matrix(TS[:3, :3]).as_quat()[[3, 0, 1, 2]]; newq[n] = (TS[:3, 3], q)
    pat = re.compile(rf'(<body name="{n}" pos=")([^"]*)(" quat=")([^"]*)(")'); assert len(pat.findall(xml)) == 1, n
    xml = pat.sub(lambda mm: mm.group(1) + ' '.join(f'{x:.9g}' for x in TS[:3, 3]) + mm.group(3) + ' '.join(f'{x:.9g}' for x in q) + mm.group(5), xml)
import mujoco
_m0 = mujoco.MjModel.from_xml_path(str(a.scene)); AD = {n: int(_m0.joint(n + '_free').qposadr[0]) for n in names}; assert AD == {'blue_cup': 56, 'wooden_spoon': 63}, AD
km = re.search(r'(<key name="home" qpos=")([^"]*)(")', xml); qv = km.group(2).split(); assert len(qv) == 70
qv[56:63] = [f'{x:.9g}' for x in np.r_[newq['blue_cup'][0], newq['blue_cup'][1]]]; qv[63:70] = [f'{x:.9g}' for x in np.r_[newq['wooden_spoon'][0], newq['wooden_spoon'][1]]]
xml = xml[:km.start(2)] + ' '.join(qv) + xml[km.end(2):]
(out / 'scene').mkdir(parents=True, exist_ok=True); scene = out / 'scene/scene_physics.xml'; scene.write_text(xml)
import mujoco
mm_ = mujoco.MjModel.from_xml_path(str(scene)); dd = mujoco.MjData(mm_); mujoco.mj_resetDataKeyframe(mm_, dd, 0); mujoco.mj_forward(mm_, dd)
pen = {}
for c in dd.contact[:dd.ncon]:
    b1, b2 = mm_.body(mm_.geom_bodyid[c.geom1]).name, mm_.body(mm_.geom_bodyid[c.geom2]).name; key = '|'.join(sorted([b1[:20], b2[:20]])); pen[key] = max(pen.get(key, 0), -float(c.dist) * 1000)
chk['frame0_contacts_penetration_mm'] = pen; print('frame-0 contacts (mm):', {k: round(v, 1) for k, v in pen.items()})
rec = dict(episode=EP, reference_dir=str(CMP), shipped_meshes={n: sha(Path(a.meshes) / f'{n}.obj') for n in names}, method='target_final_v5 tracks -> our body frames (kit scorer registration; cup yaw about its symmetry axis fixed to ours) -> our world via ONE rigid transform (frame-0 Kabsch + minimal tilt about the frame-0 cup base + shift putting the shipped 3c3c cup bottom and the lowest spoon point on the table); physics scene initial object poses = compose frame 0',
           target_track_sha256={n: sha(CMP / f'{n}_trajectory.npz') for n in names}, target_mesh_sha256={n: sha(CMP / f'{n}_mesh_metric.obj') for n in names}, G_target_body_in_our_body={n: G[n].tolist() for n in names},
           world_transform_X=X.tolist(), checks=chk, shift_vs_v21_reference=dict(mean_cm={n: float(shift[n].mean()) for n in names}, mean_deg={n: float(rot[n].mean()) for n in names}, per_frame_cm={n: shift[n].round(3).tolist() for n in names}, per_frame_deg={n: rot[n].round(2).tolist() for n in names}),
           base_scene=str(a.scene), base_scene_sha256=sha(a.scene), scene=str(scene), scene_sha256=sha(scene), original_object_poses_sha256=sha(SRC / 'object_poses.npz'), corrected_object_poses_sha256=sha(OS / 'object_poses.npz'), script=str(Path(__file__).resolve()), script_sha256=sha(__file__))
(OS / 'target_retarget.json').write_text(json.dumps(rec, indent=1))
OE.mkdir(parents=True)
for p in EPD.iterdir(): (shutil.copytree(p, OE / p.name, symlinks=True) if p.is_dir() else shutil.copy2(p, OE / p.name))
mf = json.loads((EPD / 'input_manifest.json').read_text())
trk = dict(np.load(EPD / 'tracking_reference.npz', allow_pickle=False)); q = trk['qpos'].copy()
q[:, 56:63] = np.c_[(Sm @ W[:, ci])[:, :3, 3], R.from_matrix((Sm @ W[:, ci])[:, :3, :3]).as_quat()[:, [3, 0, 1, 2]]]
q[:, 63:70] = np.c_[(Sm @ W[:, si])[:, :3, 3], R.from_matrix((Sm @ W[:, si])[:, :3, :3]).as_quat()[:, [3, 0, 1, 2]]]
assert np.allclose(q[0, 56:63], [float(x) for x in qv[56:63]], atol=1e-6) and np.allclose(q[0, 63:70], [float(x) for x in qv[63:70]], atol=1e-6)
trk['qpos'] = q; np.savez(OE / 'tracking_reference.npz', **trk)
rt = dict(np.load(EPD / 'reference_trajectory.npz', allow_pickle=False)); rt['qpos'] = rt['qpos'].copy(); rt['qpos'][:, 56:70] = q[:, 56:70]
rt['object_positions'] = rt['object_positions'].copy(); rt['object_quaternions_wxyz'] = rt['object_quaternions_wxyz'].copy()
for i, n in enumerate([str(x) for x in rt['object_names']]):
    TS = Sm @ W[:, zn.index(n)]; rt['object_positions'][:, i] = TS[:, :3, 3]; rt['object_quaternions_wxyz'][:, i] = R.from_matrix(TS[:, :3, :3]).as_quat()[:, [3, 0, 1, 2]]
np.savez(OE / 'reference_trajectory.npz', **rt)
mf.update(source=str(OS), source_reference=str(OS / 'reference.parquet'), source_reference_sha256=sha(OS / 'reference.parquet'), source_object_poses_sha256=sha(OS / 'object_poses.npz'))
mf['target_retarget'] = dict(record=str(OS / 'target_retarget.json'), record_sha256=sha(OS / 'target_retarget.json'), script=str(Path(__file__).resolve()), script_sha256=sha(__file__), scene=str(scene), scene_sha256=sha(scene),
    base_episode_dir=str(EPD), base_manifest_sha256=sha(EPD / 'input_manifest.json'))
(OE / 'input_manifest.json').write_text(json.dumps(mf, indent=2) + '\n'); print('wrote', out)
