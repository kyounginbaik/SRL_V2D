#!/usr/bin/env python3
"""Stream M phase 6 (v21C hedge, episode 14): re-target the episode-14 object reference to the coworker's compose_eval poses while
keeping OUR meshes (rack 090b7d17 six-prong, cup 78bd065c).

1. Body conversion (exactly what the kit scorer does): for each object, the kit registration of OUR submitted mesh (decoded from the
   v21 submission parquet, i.e. the packer's reduced mesh) onto the compose mesh (decimated as in score_vs_compose.py) with the
   scorer seed 1000003*ep + 101*slot gives (candidate_anchor ca, rotation rot). The scorer compares X_ours converted
   (p + R ca, R rot^T) with X_compose anchored (p + R anchor, R). Hence the compose pose expressed in our body frame is
   X_ours = X_compose @ G,  G = [[rot, anchor - rot ca], [0, 1]].
2. World placement: ONE rigid transform W for the whole episode. The compose rack is static (held static for the clip); W maps
   the compose rack (in our body frame) exactly onto our v21 rack pose at frame 0, so the rack sits on our table exactly as in the
   v21 world and the cup follows the compose cup-relative-to-rack track. The kit's frame-0 scene alignment removes any common rigid
   transform, so W does not change the score. The camera-chain transform (world_from_rgb[0] A_T_rect inv(cam0_T_cam[0])) is
   reported for comparison.
3. Self-check: the new reference, scored against compose with the kit functions (aligned_errors), must give ~0 cm / ~0 deg.
4. Writes a corrected copy of the source dir (object_poses.npz, reference.parquet object columns, <obj>_trajectory.npz) and
   compose_target_correction.json (hashes, transforms, registration residuals, per-frame shift vs the v21 reference).
usage: compose_reference.py SRC_DIR OUT_DIR"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import sys, json, hashlib, shutil
from pathlib import Path
import numpy as np, pandas as pd, trimesh, pyarrow as pa, pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, '/data1/SRL_V2D/challenge_submissions_SRL/analysis_20261020/scripts')
import public_kit_score as K
EP = 14; NAMES = ['blue_cup', 'plastic_dish_rack']   # slot order of the episode-14 results parquet
C16 = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023'); CMP = C16 / 'compose_target' / f'episode_{EP:06d}'
SUB = Path('/data1/SRL_V2D/challenge_submissions/track3_v21_20261024/all_20_episodes_post/track3_submission.parquet')
STEREO = Path('/data1/SRL_V2D/challenge_reconstruction/eval_stereo_current_20261010/episode_000014/rectification.npz')
CAL = json.load(open('/data1/video_to_data_challenge/track_3/public/meta/camera_calibration.json'))['cameras']
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
M = K.load_metric('MP-SR')


def ref_geom(n):   # identical to score_vs_compose.ref_geometry for one object
    m = trimesh.load(CMP / f'{n}_mesh_metric.obj', force='mesh'); m.merge_vertices()
    if len(m.faces) > 40000:
        try: m = m.simplify_quadric_decimation(face_count=40000)
        except Exception: pass
    m.remove_unreferenced_vertices(); v = np.asarray(m.vertices, np.float64); f = np.asarray(m.faces, np.int64)
    a, _ = M._surface_moments(v, f); return v, f, a


def T(p, q_xyzw):
    X = np.tile(np.eye(4), (len(p), 1, 1)); X[:, :3, :3] = R.from_quat(q_xyzw).as_matrix(); X[:, :3, 3] = p; return X


def main():
    src, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    if out.exists(): raise SystemExit(f'exists: {out}')
    sub = pd.read_parquet(SUB); sub = sub[sub.row_id.str.contains(f'/e{EP:06d}/') | sub.row_id.str.contains(f'/e{EP}/')]
    if not len(sub): raise SystemExit('episode rows not found in v21 parquet')
    z = dict(np.load(src / 'object_poses.npz')); names = [str(x) for x in z['names']]; Wv21 = z['world_from_object'].astype(float); Nf = len(Wv21)
    G = {}; reg = {}
    for slot, n in enumerate(NAMES):
        rows = sub[sub.row_id.str.endswith(f'/o{slot}') | sub.row_id.str.contains(f'/o{slot}$')]
        enc = rows.object_mesh_b64.fillna('').astype(str); enc = enc[enc.str.strip() != '']
        cand = M._decode_object_mesh(enc.iloc[0]); rv, rf, anchor = ref_geom(n)
        ca, rot = M._register_object_body(cand, (rv, rf), anchor, 1_000_003 * EP + 101 * slot)
        g = np.eye(4); g[:3, :3] = rot; g[:3, 3] = anchor - rot @ ca; G[n] = g
        # residual: our (packed) mesh mapped into the compose body frame vs the compose surface
        cv = np.asarray(cand[0], float); mapped = cv @ g[:3, :3].T + g[:3, 3]
        from scipy.spatial import cKDTree; dense = M._sample_mesh_surface(rv, rf, 200000, 0); d = cKDTree(dense).query(mapped)[0]   # vertex -> nearest of 200k compose surface samples
        reg[n] = dict(G=g, candidate_vertices=int(len(cv)), residual_mm_mean=float(d.mean() * 1000), residual_mm_p95=float(np.percentile(d, 95) * 1000),
                      rotation_vs_identity_deg=float(np.degrees(R.from_matrix(rot).magnitude())), seed=1_000_003 * EP + 101 * slot)
    comp = {n: np.load(CMP / f'{n}_trajectory.npz')['T_world_obj'].astype(float) for n in NAMES}
    for n in NAMES: assert len(comp[n]) == Nf, (n, len(comp[n]), Nf)
    Y = {n: comp[n] @ G[n] for n in NAMES}
    rk = 'plastic_dish_rack'; rack_motion = float(np.max(np.linalg.norm(Y[rk][:, :3, 3] - Y[rk][0, :3, 3], axis=1)) * 100)
    Wst = Wv21[0, names.index(rk)] @ np.linalg.inv(Y[rk][0])
    # camera-chain transform for comparison
    r = np.load(STEREO); Rm = np.eye(4); Rm[:3, :3] = r['R1'].T
    T_AC = np.linalg.inv(np.array(CAL['ego_cam_a']['extrinsics_to_stereo_left'])) @ np.array(CAL['ego_cam_c']['extrinsics_to_stereo_left'])
    cam0 = np.load(CMP / 'camera_motion.npz')['cam0_T_cam'].astype(float)
    Wcam = z['world_from_rgb'][0].astype(float) @ T_AC @ Rm @ np.linalg.inv(cam0[0])
    D = np.linalg.inv(Wcam) @ Wst
    new = np.stack([Wst @ Y[n] for n in NAMES], 1)          # (T, 2, 4, 4) in NAMES order
    new[:, NAMES.index(rk)] = new[0, NAMES.index(rk)]       # rack exactly static (compose holds it static; max motion recorded)
    order = [NAMES.index(n) for n in names]; newz = new[:, order]
    # self-check with the kit alignment: achieved = new (our body), reference = compose
    def kitcheck(Wobj):
        ach = np.zeros((Nf, 1, 2, 7)); ref = np.zeros((Nf, 1, 2, 7)); axes = []; orders = []
        for s, n in enumerate(NAMES):
            X = Wobj[:, names.index(n)]; ach[:, 0, s, :3] = X[:, :3, 3]; ach[:, 0, s, 3:] = R.from_matrix(X[:, :3, :3]).as_quat()
            ref[:, 0, s, :3] = comp[n][:, :3, 3]; ref[:, 0, s, 3:] = R.from_matrix(comp[n][:, :3, :3]).as_quat()
            g = G[n]; rot = g[:3, :3]; rv, rf, anchor = ref_geom(n); ca = rot.T @ (anchor - g[:3, 3])
            ach[:, :, s] = M._convert_object_body(ach[:, :, s], ca, rot)
            ref[:, :, s, :3] += np.einsum('...ij,j->...i', M._rotation_matrix(ref[:, :, s, 3:]), anchor)
            ax, o = K.symmetry_axis(n, rv); axes.append(ax); orders.append(int(o))
        a2 = M._align_registered_scene(ach, ref, axes)
        pos = np.linalg.norm(a2[..., :3] - ref[..., :3], axis=-1).mean(axis=0)[0] * 100
        ori = np.degrees(M._symmetry_orientation_errors(a2, ref, axes, orders).mean(axis=0)[0])
        return {n: (round(float(pos[i]), 3), round(float(ori[i]), 2)) for i, n in enumerate(NAMES)}
    check_new = kitcheck(newz); check_v21 = kitcheck(Wv21)
    shift = {n: dict(pos_cm=np.linalg.norm(newz[:, names.index(n), :3, 3] - Wv21[:, names.index(n), :3, 3], axis=1) * 100,
                     rot_deg=np.degrees(R.from_matrix(np.einsum('tji,tjk->tik', Wv21[:, names.index(n), :3, :3], newz[:, names.index(n), :3, :3])).magnitude())) for n in names}
    # write the corrected source
    shutil.copytree(src, out, ignore=shutil.ignore_patterns('*.mp4', '__pycache__'))
    z['world_from_object'] = newz; np.savez_compressed(out / 'object_poses.npz', **z)
    for n in names:
        p = src / f'{n}_trajectory.npz'
        if p.exists():
            tr = dict(np.load(p)); X = newz[:, names.index(n)]; tr['T_world_obj'] = X
            if 'T_rgb_obj' in tr: tr['T_rgb_obj'] = np.linalg.inv(z['world_from_rgb']) @ X
            np.savez_compressed(out / f'{n}_trajectory.npz', **tr)
    tab = pq.read_table(src / 'reference.parquet'); row = tab.to_pylist()[0]; bn = list(row['object_body_names'])
    pos = np.array(row['object_body_position']); q = np.array(row['object_body_wxyz'])
    for j, n in enumerate(bn):
        X = newz[:, names.index(n)]; pos[:, j] = X[:, :3, 3]; qq = R.from_matrix(X[:, :3, :3]).as_quat(); q[:, j] = qq[:, [3, 0, 1, 2]]
    j0 = bn.index(row['object_name']); X0 = newz[:, names.index(row['object_name'])]
    row['object_root_position'] = X0[:, :3, 3].tolist(); row['object_root_axis_angle'] = R.from_matrix(X0[:, :3, :3]).as_rotvec().tolist()
    row['object_body_position'] = pos.tolist(); row['object_body_wxyz'] = q.tolist()
    pq.write_table(pa.Table.from_pylist([row], schema=tab.schema), out / 'reference.parquet')
    tl = {n: {f'{t/20:.1f}s': (round(float(shift[n]['pos_cm'][t]), 2), round(float(shift[n]['rot_deg'][t]), 1)) for t in range(0, Nf, 10)} for n in names}
    rec = dict(schema='compose_target_correction_v1', episode=EP, purpose='v21C hedge: object reference re-targeted to the coworker compose_eval poses, our meshes kept',
               compose_dir=str(CMP), compose_tracks={n: dict(path=str(CMP / f'{n}_trajectory.npz'), sha256=sha(CMP / f'{n}_trajectory.npz')) for n in NAMES},
               compose_meshes={n: dict(path=str(CMP / f'{n}_mesh_metric.obj'), sha256=sha(CMP / f'{n}_mesh_metric.obj'), use='registration target only (never in our world)') for n in NAMES},
               v21_submission=dict(path=str(SUB), sha256=sha(SUB), use='our packed meshes for the exact kit registration'),
               registration=reg, world_transform_static_rack=Wst, world_transform_camera_chain=Wcam,
               static_vs_camera_chain=dict(translation_cm=float(np.linalg.norm(D[:3, 3]) * 100), rotation_deg=float(np.degrees(R.from_matrix(D[:3, :3]).magnitude()))),
               compose_rack_max_motion_cm=rack_motion, rack_made_exactly_static=True,
               kit_selfcheck_new_vs_compose=check_new, kit_v21_reference_vs_compose=check_v21,
               shift_vs_v21_reference_mean={n: (float(shift[n]['pos_cm'].mean()), float(shift[n]['rot_deg'].mean())) for n in names},
               shift_vs_v21_reference_timeline=tl,
               original_object_poses_sha256=sha(src / 'object_poses.npz'), corrected_object_poses_sha256=sha(out / 'object_poses.npz'),
               original_reference_parquet_sha256=sha(src / 'reference.parquet'), corrected_reference_parquet_sha256=sha(out / 'reference.parquet'),
               source_dir=str(src))
    (out / 'compose_target_correction.json').write_text(json.dumps(rec, indent=1, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)) + '\n')
    print(json.dumps({k: rec[k] for k in ('static_vs_camera_chain', 'compose_rack_max_motion_cm', 'kit_selfcheck_new_vs_compose', 'kit_v21_reference_vs_compose', 'shift_vs_v21_reference_mean')}, indent=1, default=str))
    print(json.dumps({n: {k: v for k, v in r.items() if k != 'G'} for n, r in reg.items()}, indent=1)); print(json.dumps(tl))


if __name__ == '__main__': main()
