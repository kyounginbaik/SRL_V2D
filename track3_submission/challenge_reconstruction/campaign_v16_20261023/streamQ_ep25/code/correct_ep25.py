#!/usr/bin/env python3
"""Stream Q, episode 25: minimal physics-consistency correction of OUR reconstructed board reference (object_poses.npz of the
slots_v3 source = user ZIP poses, sha 3369426f...). Writes a corrected copy of the source dir + board_track_correction.json.

Corrections (all rigid, per board, smoothstep blends; every other frame/column byte-identical):
 C1 "flat at rest" (start): board 1 frames 0-31 (0-1.55 s, static until the left hand pushes it), board 2 frames 0-71 (0-3.55 s,
    static until the right hand nudges it). The reconstructed boards lie tilted 5.8 / 3.8 deg against the stereo table plane
    (one end ~5-8 mm into the table, the other end up); stereo plane fits of the board faces give 0.7-1.3 deg, the independent
    GT-validated board tracker (recon_v15, public eps 23/24 ~2 cm / 4 deg) 1.0-1.6 deg, and the physics scene drops both boards
    flat within 2 frames (4.5 / 4.3 deg jump of the frame-0 state). Correction: minimal rotation that makes the face normal
    vertical (yaw kept, about the slab centre), then the vertical shift that rests the board's own collision geometry on z=0.
    Blend out over the next 7 frames (0.35 s) before the hand contact.
 C2 "standing on the table": board 1 frames 140-355 (7.0-17.75 s: stood on edge on the table at 6.75-7.0 s and never lifted again),
    board 2 frames 295-355 (14.75-17.75 s: seated in the slot on the table). Stereo: the boards' top edges stay at the resting
    height above the stereo table (126-133 mm, same value as the released cross at 16.5-17.75 s) and the two top edges are flush
    (-0.3..+1.2 cm) after seating, whereas the reference holds board 1 0.8-2.0 cm above that and 1.4-2.4 cm above board 2.
    Correction: pure vertical shift so the board's collision geometry rests on z=0 (the physics table). Blend in over 0.5 s / 0.25 s.
usage: correct_ep25.py --source SRC_EPISODE_DIR --output OUT_EPISODE_DIR [--evidence JSON ...]"""
import argparse, hashlib, json, shutil
from pathlib import Path
import numpy as np, trimesh, pyarrow as pa, pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R
COLL = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/ep25_v15/collision')   # collision of the physics scene
NAMES = ['wooden_piece_1', 'wooden_piece_2']
C1 = {'wooden_piece_1': (0, 31, 38), 'wooden_piece_2': (0, 71, 78)}            # full start..end, blend-out end
C2 = {'wooden_piece_1': (130, 140, 355), 'wooden_piece_2': (290, 295, 355)}     # blend-in start, full start..end
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
def smooth(u): u = np.clip(u, 0, 1); return u * u * (3 - 2 * u)

def frames(src):
    out = {}
    for n in NAMES:
        v = np.asarray(trimesh.load(src / f'assets/{n}_visual.obj', force='mesh').vertices); c0 = v.mean(0); w, U = np.linalg.eigh((v - c0).T @ (v - c0))
        if np.linalg.det(U) < 0: U[:, 0] *= -1
        q = (v - c0) @ U; h, e = np.histogram(q[:, 0], bins=80); pk = np.sort(e[np.argsort(h)[-2:]] + np.diff(e)[0] / 2)
        lo = np.percentile(q, .5, 0); hi = np.percentile(q, 99.5, 0); slab_c = c0 + U @ np.array([pk.mean(), (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2])
        cv = np.vstack([trimesh.load(f, force='mesh').vertices for f in sorted((COLL / f'{n}_cut3mm').glob('part_*.obj'))])
        out[n] = dict(U=U, slab_c=slab_c, coll=cv)
    return out

def apply_delta(T, c_body, rotvec, dt, w):
    """rotate by w*rotvec about the body point c_body (world position kept) and translate by w*dt"""
    Rn = R.from_rotvec(w * rotvec).as_matrix(); cw = T[:3, :3] @ c_body + T[:3, 3]
    out = T.copy(); out[:3, :3] = Rn @ T[:3, :3]; out[:3, 3] = cw + w * dt - out[:3, :3] @ c_body; return out

def flat_delta(T, f):
    thin = T[:3, :3] @ f['U'][:, 0]; tgt = np.array([0, 0, 1.]) * np.sign(thin[2] if thin[2] != 0 else 1)
    ax = np.cross(thin, tgt); s = np.linalg.norm(ax); ang = np.arctan2(s, thin @ tgt); rv = ax / s * ang if s > 1e-12 else np.zeros(3)
    Tn = apply_delta(T, f['slab_c'], rv, np.zeros(3), 1.); low = (f['coll'] @ Tn[:3, :3].T + Tn[:3, 3])[:, 2].min()
    return rv, np.array([0, 0, -low])

def rest_delta(T, f):
    low = (f['coll'] @ T[:3, :3].T + T[:3, 3])[:, 2].min(); return np.zeros(3), np.array([0, 0, -low])

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source', type=Path, required=True); ap.add_argument('--output', type=Path, required=True); ap.add_argument('--evidence', type=Path, nargs='*', default=[])
    a = ap.parse_args(); src, dst = a.source.resolve(), a.output.resolve()
    if dst.exists(): raise SystemExit('exists, never overwrite: ' + str(dst))
    z = dict(np.load(src / 'object_poses.npz')); T0 = z['world_from_object']; T = T0.copy(); names = [str(x) for x in z['names']]; assert names == NAMES
    F = frames(src); rec = dict(per_frame={}, windows={})
    for i, n in enumerate(NAMES):
        f = F[n]; s, e, b = C1[n]; rows = []
        for k in range(s, e + 1):
            rv, dt = flat_delta(T0[k, i], f); T[k, i] = apply_delta(T0[k, i], f['slab_c'], rv, dt, 1.); rows.append((k, 1., rv, dt))
        rv_e, dt_e = flat_delta(T0[e, i], f)
        for k in range(e + 1, b + 1):
            w = 1 - smooth((k - e) / (b - e)); T[k, i] = apply_delta(T0[k, i], f['slab_c'], rv_e, dt_e, w); rows.append((k, w, rv_e, dt_e))
        bs, s2, e2 = C2[n]
        rv_s, dt_s = rest_delta(T0[s2, i], f)
        for k in range(bs, s2):
            w = smooth((k - bs) / (s2 - bs)); T[k, i] = apply_delta(T0[k, i], f['slab_c'], rv_s, dt_s, w); rows.append((k, w, rv_s, dt_s))
        for k in range(s2, e2 + 1):
            rv, dt = rest_delta(T0[k, i], f); T[k, i] = apply_delta(T0[k, i], f['slab_c'], rv, dt, 1.); rows.append((k, 1., rv, dt))
        rec['windows'][n] = dict(C1_flat_full=[s / 20, e / 20], C1_blend_out=[e / 20, b / 20], C2_rest_blend_in=[bs / 20, s2 / 20], C2_rest_full=[s2 / 20, e2 / 20])
        pf = {}
        for k, w, rv, dt in rows:
            dpos = np.linalg.norm((T[k, i, :3, :3] @ f['slab_c'] + T[k, i, :3, 3]) - (T0[k, i, :3, :3] @ f['slab_c'] + T0[k, i, :3, 3]))
            drot = np.degrees((R.from_matrix(T[k, i, :3, :3]) * R.from_matrix(T0[k, i, :3, :3]).inv()).magnitude())
            pf[str(k)] = dict(weight=round(float(w), 4), slab_centre_shift_cm=round(float(dpos * 100), 3), rotation_deg=round(float(drot), 3), dz_cm=round(float(w * dt[2] * 100), 3))
        rec['per_frame'][n] = pf
    # rigidity check, frame-0 table contact
    assert np.allclose(np.swapaxes(T[..., :3, :3], -1, -2) @ T[..., :3, :3], np.eye(3), atol=1e-9)
    changed = {n: int(np.sum(np.any(np.abs(T[:, i] - T0[:, i]) > 1e-12, axis=(1, 2)))) for i, n in enumerate(NAMES)}
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns('__pycache__'))
    z['world_from_object'] = T; np.savez_compressed(dst / 'object_poses.npz', **z)
    for i, n in enumerate(NAMES):
        tr = dict(np.load(src / f'{n}_trajectory.npz')); Two = tr['T_world_obj'].copy(); Two[:] = T[:, i]
        tr['T_world_obj'] = Two
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
    # summary of the reference shift (all frames incl. unchanged ones -> means over the episode, and over corrected frames)
    summ = {}
    for i, n in enumerate(NAMES):
        f = F[n]; c0w = np.einsum('kij,j->ki', T0[:, i, :3, :3], f['slab_c']) + T0[:, i, :3, 3]; c1w = np.einsum('kij,j->ki', T[:, i, :3, :3], f['slab_c']) + T[:, i, :3, 3]
        dp = np.linalg.norm(c1w - c0w, axis=1) * 100; dr = np.degrees((R.from_matrix(T[:, i, :3, :3]) * R.from_matrix(T0[:, i, :3, :3]).inv()).magnitude()); m = dp > 1e-9
        summ[n] = dict(frames_changed=changed[n], mean_shift_cm_all_frames=float(dp.mean()), mean_rot_deg_all_frames=float(dr.mean()),
                       mean_shift_cm_changed=float(dp[m].mean()) if m.any() else 0., max_shift_cm=float(dp.max()), max_rot_deg=float(dr.max()))
    info = dict(schema='ep25_board_track_correction_v1', episode=25, script=str(Path(__file__).resolve()), script_sha256=sha(__file__),
                source=str(src), output=str(dst), collision_used_for_table_contact=str(COLL), windows=rec['windows'], reference_shift=summ,
                original_object_poses_sha256=sha(src / 'object_poses.npz'), corrected_object_poses_sha256=sha(dst / 'object_poses.npz'),
                original_reference_parquet_sha256=sha(src / 'reference.parquet'), corrected_reference_parquet_sha256=sha(dst / 'reference.parquet'),
                evidence={str(p): sha(p) for p in a.evidence}, per_frame=rec['per_frame'],
                not_corrected=['in-air segments (board 1 2.0-6.75 s, board 2 3.9-14.5 s): no reliable independent height evidence',
                               'world table plane offset (stereo table at +9..14 mm, 1-2 deg vs world z) outside the resting windows: needs a re-expressed world incl. hands',
                               'final relative pose of the two boards: already a consistent seated cross (long axes 90 deg, centres within 0.7 cm)'])
    (dst / 'board_track_correction.json').write_text(json.dumps(info, indent=1))
    print(json.dumps(dict(windows=rec['windows'], reference_shift=summ, corrected_sha=info['corrected_object_poses_sha256']), indent=1))

if __name__ == '__main__': main()
