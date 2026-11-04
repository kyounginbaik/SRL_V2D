#!/usr/bin/env python3
"""Stream Q, episode 25, correction v2 (supersedes correct_ep25.py = v1/world w1, whose board-2 start correction was contradicted by
stereo: it lifted board 2 onto its mesh's corner protrusions, 1 cm above the stereo top face). Minimal physics-consistency correction
of OUR reconstructed board reference (slots_v3 source = user ZIP poses, sha 3369426f...). Writes a corrected copy of the source dir
+ board_track_correction.json. Every frame/column outside the windows below is byte-identical.

 C1 board 1 lies flat (frames 0-47 = 0-2.35 s: resting, then slid along the table by the left hand until the lift at ~2.5 s):
    reconstructed face 4.4-5.0 deg off the stereo face plane (99 % inliers; stereo face 0.9-1.4 deg from the stereo table), the
    independent board tracker recon_v15 1.1-1.2 deg, physics drops it flat within 2 frames (4.5 deg jump of the frame-0 state).
    Minimal rotation making the face normal vertical (yaw kept, about the slab centre) + vertical shift resting the board's own
    collision geometry on z=0 (= the physics' settled state of the old scene within 0.6 mm / 0.06 deg). Blend out 2.35-2.6 s.
 C2 board 1 stands on its edge on the table (frames 140-355 = 7.0-17.75 s): stereo top edge stays at the resting height of the
    released cross (126-133 mm above the stereo table) while the reference holds it 0.8-2.0 cm higher (and 1.2-2.2 cm above the
    physics table). Pure vertical shift resting its collision on z=0. Blend in 6.5-7.0 s.
 C3 board 2 engaged in board 1's slot (frames 230-355 = 11.5-17.75 s): with the slot-cut collision of the physics scene the
    reconstructed pair interpenetrates by 5-16.5 mm during 11.25-16.5 s (3.5 mm at rest), i.e. board 2 is not where an engaged
    board can be; stereo fits board 1 (median 0.8-1.6 mm) better than board 2 (5-9 mm). Board 2 := board 1 (corrected) x the
    geometric assembly transform (long axes perpendicular, mid axes parallel with the signs of the final cross, board-2 slot centre
    on board 1's plate mid-plane, board-2 plate mid-plane on board 1's slot centre) with the reconstructed per-frame insertion depth
    (board-2 centre along board 1's mid axis, which matches the stereo top-edge difference). From 14.75 s (stereo: top edges flush,
    seated) board 2 rests on z=0. Blend in 11.0-11.5 s (vertical rest blend 14.5-14.75 s).
usage: correct_ep25_v2.py --source SRC_EPISODE_DIR --output OUT_EPISODE_DIR [--evidence FILE ...]"""
import argparse, hashlib, json, shutil
from pathlib import Path
import numpy as np, trimesh, pyarrow as pa, pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R, Slerp
COLL = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/ep25_v15/collision')
SLOT = json.loads((COLL / 'cut_report_cut3mm.json').read_text())
NAMES = ['wooden_piece_1', 'wooden_piece_2']
C1 = (0, 47, 52); C2 = (130, 140, 355); C3 = (220, 230, 355); C3_REST = (290, 295)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
def smooth(u): u = np.clip(u, 0, 1); return u * u * (3 - 2 * u)

def frames(src):
    out = {}
    for n in NAMES:
        v = np.asarray(trimesh.load(src / f'assets/{n}_visual.obj', force='mesh').vertices); c0 = v.mean(0); w, U = np.linalg.eigh((v - c0).T @ (v - c0))
        if np.linalg.det(U) < 0: U[:, 0] *= -1        # same convention as build_slot_collision.py / cut_slot_parts.py
        q = (v - c0) @ U; h, e = np.histogram(q[:, 0], bins=80); pk = np.sort(e[np.argsort(h)[-2:]] + np.diff(e)[0] / 2)
        lo = np.percentile(q, .5, 0); hi = np.percentile(q, 99.5, 0); slab_c = c0 + U @ np.array([pk.mean(), (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2])
        cv = np.vstack([trimesh.load(f, force='mesh').vertices for f in sorted((COLL / f'{n}_cut3mm').glob('part_*.obj'))])
        qc = (cv - c0) @ U; sl = 0.5 * sum(SLOT[n]['slot_long_cm']) / 100; band = np.abs(qc[:, 2] - sl) < .015
        tc = 0.5 * (qc[band, 0].max() + qc[band, 0].min())                    # plate thin centre at the crossing (collision)
        out[n] = dict(c0=c0, U=U, slab_c=slab_c, coll=cv, slot_long=sl, plate_thin=tc)
    return out

def apply_delta(T, c_body, rotvec, dt, w):
    Rn = R.from_rotvec(w * rotvec).as_matrix(); cw = T[:3, :3] @ c_body + T[:3, 3]
    out = T.copy(); out[:3, :3] = Rn @ T[:3, :3]; out[:3, 3] = cw + w * dt - out[:3, :3] @ c_body; return out
def lowest(T, f): return (f['coll'] @ T[:3, :3].T + T[:3, 3])[:, 2].min()
def flat_delta(T, f):
    thin = T[:3, :3] @ f['U'][:, 0]; tgt = np.array([0, 0, 1.]) * np.sign(thin[2])
    ax = np.cross(thin, tgt); s = np.linalg.norm(ax); rv = ax / s * np.arctan2(s, thin @ tgt) if s > 1e-12 else np.zeros(3)
    Tn = apply_delta(T, f['slab_c'], rv, np.zeros(3), 1.); return rv, np.array([0, 0, -lowest(Tn, f)])
def blend_pose(Ta, Tb, w):
    out = Ta.copy(); out[:3, :3] = Slerp([0, 1], R.from_matrix([Ta[:3, :3], Tb[:3, :3]]))(w).as_matrix(); out[:3, 3] = (1 - w) * Ta[:3, 3] + w * Tb[:3, 3]; return out

def assembly(F, T1, T2):
    """body-2 -> body-1 transform of the ideal engaged pair, insertion depth taken from (T1,T2); returns X (4x4) and the signed permutation"""
    f1, f2 = F[NAMES[0]], F[NAMES[1]]
    M = f1['U'].T @ T1[:3, :3].T @ T2[:3, :3] @ f2['U']                 # b2 PCA axes in b1 PCA coordinates
    P = np.zeros((3, 3))
    for j in range(3): i = int(np.argmax(np.abs(M[:, j]))); P[i, j] = np.sign(M[i, j])
    assert sorted(np.argmax(np.abs(P), 0).tolist()) == [0, 1, 2] and np.argmax(np.abs(P[:, 1])) == 1, P   # mid axes parallel, long<->thin swapped
    Xr = f1['U'] @ P @ f2['U'].T
    c2_in_1 = T1[:3, :3].T @ (T2[:3, :3] @ f2['c0'] + T2[:3, 3] - T1[:3, 3]); depth = (f1['U'].T @ (c2_in_1 - f1['c0']))[1]   # b2 centre along b1 mid
    s2 = np.array([f2['plate_thin'], 0., f2['slot_long']])                 # b2 slot centre on its plate (b2 PCA)
    tgt1 = np.array([f1['plate_thin'], depth, f1['slot_long']])            # b1 plate mid-plane, depth, b1 slot centre (b1 PCA)
    Xt = f1['c0'] + f1['U'] @ tgt1 - Xr @ (f2['c0'] + f2['U'] @ s2)
    X = np.eye(4); X[:3, :3] = Xr; X[:3, 3] = Xt; return X, P

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source', type=Path, required=True); ap.add_argument('--output', type=Path, required=True); ap.add_argument('--evidence', type=Path, nargs='*', default=[])
    a = ap.parse_args(); src, dst = a.source.resolve(), a.output.resolve()
    if dst.exists(): raise SystemExit('exists, never overwrite: ' + str(dst))
    z = dict(np.load(src / 'object_poses.npz')); T0 = z['world_from_object']; T = T0.copy(); assert [str(x) for x in z['names']] == NAMES
    F = frames(src); f1, f2 = F[NAMES[0]], F[NAMES[1]]; W = {n: {} for n in NAMES}
    # C1 board 1
    s, e, b = C1
    for k in range(s, e + 1): rv, dt = flat_delta(T0[k, 0], f1); T[k, 0] = apply_delta(T0[k, 0], f1['slab_c'], rv, dt, 1.); W[NAMES[0]][k] = 1.
    rv_e, dt_e = flat_delta(T0[e, 0], f1)
    for k in range(e + 1, b + 1): w = 1 - smooth((k - e) / (b - e)); T[k, 0] = apply_delta(T0[k, 0], f1['slab_c'], rv_e, dt_e, w); W[NAMES[0]][k] = w
    # C2 board 1
    bs, s2, e2 = C2; dz = -lowest(T0[s2, 0], f1)
    for k in range(bs, s2): w = smooth((k - bs) / (s2 - bs)); T[k, 0] = apply_delta(T0[k, 0], f1['slab_c'], np.zeros(3), np.array([0, 0, dz]), w); W[NAMES[0]][k] = w
    for k in range(s2, e2 + 1): T[k, 0] = apply_delta(T0[k, 0], f1['slab_c'], np.zeros(3), np.array([0, 0, -lowest(T0[k, 0], f1)]), 1.); W[NAMES[0]][k] = 1.
    # C3 board 2 (relative to the corrected board 1)
    bs3, s3, e3 = C3; Xfin, P = assembly(F, T0[350, 0], T0[350, 1]); recs = {}
    def engaged(k):
        X, _ = assembly(F, T0[k, 0], T0[k, 1]); Tn = T[k, 0] @ X
        # depth was measured relative to the ORIGINAL board 1; T[k,0] may be lowered (C2): keep the original relative depth
        return Tn
    for k in range(bs3, e3 + 1):
        Tn = engaged(k)
        if k >= C3_REST[0]:
            wr = smooth((k - C3_REST[0]) / (C3_REST[1] - C3_REST[0])); Tn = Tn.copy(); Tn[2, 3] += wr * (-lowest(Tn, f2))
        w = 1. if k >= s3 else smooth((k - bs3) / (s3 - bs3)); T[k, 1] = blend_pose(T0[k, 1], Tn, w); W[NAMES[1]][k] = w
    assert np.allclose(np.swapaxes(T[..., :3, :3], -1, -2) @ T[..., :3, :3], np.eye(3), atol=1e-9)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns('__pycache__'))
    z['world_from_object'] = T; np.savez_compressed(dst / 'object_poses.npz', **z)
    for i, n in enumerate(NAMES):
        tr = dict(np.load(src / f'{n}_trajectory.npz')); Two = tr['T_world_obj'].copy(); Two[:] = T[:, i]; tr['T_world_obj'] = Two
        if 'T_rgb_obj' in tr: tr['T_rgb_obj'] = (np.linalg.inv(z['world_from_rgb'].astype(float)) @ Two).astype(tr['T_rgb_obj'].dtype)
        np.savez_compressed(dst / f'{n}_trajectory.npz', **tr)
    tab = pq.read_table(src / 'reference.parquet'); row = tab.to_pylist()[0]; bn = list(row['object_body_names'])
    pos = np.array(row['object_body_position']); q = np.array(row['object_body_wxyz'])
    for i, n in enumerate(NAMES):
        j = bn.index(n); pos[:, j] = T[:, i, :3, 3]; qq = R.from_matrix(T[:, i, :3, :3]).as_quat(); q[:, j] = qq[:, [3, 0, 1, 2]]
        if j == 0 and bn[0] == row['object_name']:
            row['object_root_position'] = pos[:, 0].tolist(); row['object_root_axis_angle'] = R.from_matrix(T[:, i, :3, :3]).as_rotvec().tolist()
    row['object_body_position'] = pos.tolist(); row['object_body_wxyz'] = q.tolist()
    pq.write_table(pa.Table.from_pylist([row], schema=tab.schema), dst / 'reference.parquet')
    summ, per = {}, {}
    for i, n in enumerate(NAMES):
        f = F[n]; c0w = np.einsum('kij,j->ki', T0[:, i, :3, :3], f['slab_c']) + T0[:, i, :3, 3]; c1w = np.einsum('kij,j->ki', T[:, i, :3, :3], f['slab_c']) + T[:, i, :3, 3]
        dp = np.linalg.norm(c1w - c0w, axis=1) * 100; dr = np.degrees((R.from_matrix(T[:, i, :3, :3]) * R.from_matrix(T0[:, i, :3, :3]).inv()).magnitude()); m = (dp > 1e-9) | (dr > 1e-9)
        summ[n] = dict(frames_changed=int(m.sum()), mean_shift_cm_all_frames=float(dp.mean()), mean_rot_deg_all_frames=float(dr.mean()),
                       mean_shift_cm_changed=float(dp[m].mean()) if m.any() else 0., mean_rot_deg_changed=float(dr[m].mean()) if m.any() else 0., max_shift_cm=float(dp.max()), max_rot_deg=float(dr.max()))
        per[n] = {str(k): dict(weight=round(float(W[n].get(k, 0.)), 4), slab_centre_shift_cm=round(float(dp[k]), 3), rotation_deg=round(float(dr[k]), 3), dz_cm=round(float((c1w[k, 2] - c0w[k, 2]) * 100), 3)) for k in range(len(T)) if m[k]}
    info = dict(schema='ep25_board_track_correction_v2', episode=25, script=str(Path(__file__).resolve()), script_sha256=sha(__file__), source=str(src), output=str(dst),
                collision_used=str(COLL), slot_report=SLOT, assembly_signed_permutation_b2_in_b1=P.tolist(),
                windows=dict(wooden_piece_1=dict(C1_flat_rest_full_s=[C1[0] / 20, C1[1] / 20], C1_blend_out_s=[C1[1] / 20, C1[2] / 20], C2_rest_blend_in_s=[C2[0] / 20, C2[1] / 20], C2_rest_full_s=[C2[1] / 20, C2[2] / 20]),
                             wooden_piece_2=dict(C3_engaged_blend_in_s=[C3[0] / 20, C3[1] / 20], C3_engaged_full_s=[C3[1] / 20, C3[2] / 20], C3_rest_blend_in_s=[C3_REST[0] / 20, C3_REST[1] / 20])),
                reference_shift=summ, original_object_poses_sha256=sha(src / 'object_poses.npz'), corrected_object_poses_sha256=sha(dst / 'object_poses.npz'),
                original_reference_parquet_sha256=sha(src / 'reference.parquet'), corrected_reference_parquet_sha256=sha(dst / 'reference.parquet'),
                evidence={str(p): sha(p) for p in a.evidence}, per_frame=per,
                not_corrected=['board 2 at the start: stereo top-face height agrees with the reconstruction (1.6 mm); only a 2 deg tilt difference -> unchanged (its mesh has 1.3 cm corner protrusions on the down side, a mesh defect the physics resolves by lifting it)',
                               'in-air segments (board 1 2.6-6.5 s, board 2 0-11.0 s): no hard contradiction / no reliable independent height evidence',
                               'world table plane offset (stereo table +9..14 mm, 1-2 deg vs world z) outside the resting windows: would need a re-expressed world incl. hands'])
    (dst / 'board_track_correction.json').write_text(json.dumps(info, indent=1))
    print(json.dumps(dict(windows=info['windows'], reference_shift=summ, P=P.tolist(), corrected_sha=info['corrected_object_poses_sha256']), indent=1))

if __name__ == '__main__': main()
