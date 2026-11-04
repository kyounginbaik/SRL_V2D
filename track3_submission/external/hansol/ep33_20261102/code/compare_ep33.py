#!/usr/bin/env python3
"""ep33 in our RGB camera: our HGA spoon/cup target vs the v31a and v29 robot rollouts (recorded physics).
Every robot track is aligned onto the HGA target exactly as the official scorer does (kit mesh registration of its own
mesh to the target mesh, then anchor frame-0 scene alignment), then expressed in the RGB camera of our HGA bundle
(world_from_rgb) and shifted onto the RGB clock (-delta). All tracks are drawn with the target meshes, so they differ
only in pose. Spoon: silhouette, long axis with a dot on the bowl end, and a short tick along one fixed body axis across
the bowl (it flips side when the spoon rolls over). Per frame: kit spoon orientation error vs HGA, the angle between the
long axes (end-for-end turn) and the roll about the long axis. No GT is read.
usage: compare_ep33.py OUT_PREFIX [--fps 10]"""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd
import trimesh

sys.path.insert(0, '/data1/SRL_V2D/challenge_submissions_SRL/analysis_20261020/scripts')
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG15/code')
sys.path.insert(0, '/data1/hansol/mink_S1_20261031/code')
import public_kit_score as K  # noqa: E402
from retime import shift  # noqa: E402
from contact_smooth import DELTA, arg  # noqa: E402

EP = 33
SUBS = dict(v31a=Path('/data1/SRL_V2D/challenge_submissions_SRL/track3_v31a_HGA_20261101/all_20_episodes'),
            v29=Path('/data1/SRL_V2D/challenge_submissions_SRL/track3_v29_20261030/all_20_episodes'))
REF = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/target_E14_HGA/episode_000033')
BUNDLE = Path('/home/hansol/v2d/outputs/variants_20261030/recon_E14_best_B2A_S1_HGA/episode_000033')
CAN = Path('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/canonical/evaluation/episode_000033')
COL = dict(HGA=(60, 230, 60), v31a=(230, 60, 220), v29=(0, 165, 255))  # BGR
SHEET = list(range(158, 182, 2))


def load_mesh(p):
    m = trimesh.load(p, force='mesh'); m.merge_vertices()
    if len(m.faces) > 40000:
        try:
            m = m.simplify_quadric_decimation(face_count=40000)
        except Exception:
            pass
    m.remove_unreferenced_vertices(); return np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64)


def spoon_axes(v):
    """Body-frame long axis (unit, pointing to the bowl end), the two ends, and a fixed cross axis (smallest variance)."""
    c = v.mean(0); w, U = np.linalg.eigh(np.cov((v - c).T)); ax, cross = U[:, 2], U[:, 0]
    t = (v - c) @ ax; lo, hi = np.quantile(t, [0.15, 0.85])
    def spread(sel):
        q = v[sel] - c; q = q - np.outer(q @ ax, ax); return np.linalg.norm(q, axis=1).mean()
    if spread(t < lo) > spread(t > hi):
        ax = -ax; t = -t
    return c, ax, cross, c + ax * t.max(), c + ax * t.min()


def main():
    pre = sys.argv[1]; fps = arg('--fps', 10)
    M = K.load_metric('MP-SR')
    res = pd.read_parquet(SUBS['v31a'] / 'results' / f'episode_{EP:06d}.parquet')
    names = [n for _, n in res.drop_duplicates('object_id').sort_values('object_id')[['object_id', 'object_body_name']].values.tolist()]
    tracks = [np.load(REF / f'{n}_trajectory.npz')['T_world_obj'] for n in names]
    geo = {}
    for i, n in enumerate(names):
        v, fc = load_mesh(REF / f'{n}_mesh_metric.obj'); a, _ = M._surface_moments(v, fc); ax, o = K.symmetry_axis(n, v)
        geo[i] = dict(v=v, f=fc, anchor=a, axis=ax, order=int(o))
    subs = {}
    for tag, p in SUBS.items():
        s_all = pd.read_parquet(p / 'track3_submission.parquet'); subs[tag] = s_all[s_all.row_id.str.contains(f'/e{EP:06d}/')].copy()
    rows = []
    for _, s in subs['v31a'].iterrows():
        f = int(s.row_id.split('/f')[1][:6]); slot = int(s.row_id.split('/o')[1]); c = K.pose_cols(tracks[slot][min(f, len(tracks[slot]) - 1)][None])
        rows.append(dict(row_id=s.row_id, episode_index=EP, world_index=0, frame_index=f, object_slot=slot, object_name=names[slot], **{'ref_' + k: float(v[0]) for k, v in c.items()}))
    sol = pd.DataFrame(rows)

    def aligned(sub):
        cand = []
        for slot in range(len(names)):
            enc = sub[[int(r.split('/o')[1]) == slot for r in sub.row_id]].object_mesh_b64.fillna('').astype(str)
            cand.append(M._decode_object_mesh(enc[enc.str.strip() != ''].iloc[0]))
        merged = sol.merge(sub[['row_id', *M._REQUIRED_SUBMISSION_COLUMNS]], on='row_id')
        ach, ref, _, en = M._episode_tensors(merged); assert [str(n) for n in en] == names
        ach = M._normalise_quaternions(ach, 'submission', ValueError).copy(); ref = M._normalise_quaternions(ref, 'solution', ValueError).copy()
        for slot in range(len(names)):
            ca, rot = M._register_object_body(cand[slot], (geo[slot]['v'], geo[slot]['f']), geo[slot]['anchor'], 1_000_003 * EP + 101 * slot)
            ach[:, :, slot] = M._convert_object_body(ach[:, :, slot], ca, rot)
            ref[:, :, slot, :3] += np.einsum('...ij,j->...i', M._rotation_matrix(ref[:, :, slot, 3:]), geo[slot]['anchor'])
        axes = [geo[s]['axis'] for s in range(len(names))]; orders = [geo[s]['order'] for s in range(len(names))]
        ach = M._align_registered_scene(ach, ref, axes)
        ori = np.degrees(M._symmetry_orientation_errors(ach, ref, axes, orders))[:, 0]

        def to_T(x, slot):
            Rm = M._rotation_matrix(x[:, 0, slot, 3:]); T = np.tile(np.eye(4), (len(x), 1, 1)); T[:, :3, :3] = Rm; T[:, :3, 3] = x[:, 0, slot, :3] - Rm @ geo[slot]['anchor']
            return T
        return {n: to_T(ach, s) for s, n in enumerate(names)}, {n: to_T(ref, s) for s, n in enumerate(names)}, ori

    out = {tag: aligned(sub) for tag, sub in subs.items()}
    k = names.index('wooden_spoon'); c0, lax, cross, bowl, tail = spoon_axes(geo[k]['v'])

    def turn(Ta, Tr):  # angle between long axes (deg) and roll about the long axis (deg, signed)
        a, r = Ta[:3, :3] @ lax, Tr[:3, :3] @ lax; tilt = np.degrees(np.arccos(np.clip(a @ r, -1, 1)))
        Rrel = Tr[:3, :3].T @ Ta[:3, :3]; x = Rrel @ cross; x = x - (x @ lax) * lax; x /= max(np.linalg.norm(x), 1e-9)
        roll = np.degrees(np.arctan2(np.cross(cross, x) @ lax, cross @ x)); return float(tilt), float(roll)

    z = np.load(BUNDLE / 'object_poses.npz'); Fz = len(z['world_from_rgb']); d = DELTA[f'evaluation_{EP}']; d = np.asarray(d, float)[:Fz] if np.ndim(d) else np.full(Fz, float(d))
    Wr = z['world_from_rgb'].astype(float)

    def cam(Tw):
        n_ = min(len(Tw), Fz); C = np.stack([np.linalg.inv(Wr[f]) @ Tw[f] for f in range(n_)]); return shift(C, -d[:n_])
    poses = {'HGA': {n: cam(out['v31a'][1][n]) for n in names}}
    for tag in SUBS:
        poses[tag] = {n: cam(out[tag][0][n]) for n in names}
    I = json.loads((CAN / 'intrinsics.json').read_text()); Kc = np.array([[I['fx'], 0, I['cx']], [0, I['fy'], I['cy']], [0, 0, 1.]])
    pts = {n: np.asarray(trimesh.sample.sample_surface(trimesh.Trimesh(geo[s]['v'], geo[s]['f'], process=False), 60000, seed=1)[0], float) for s, n in enumerate(names)}
    cap = cv2.VideoCapture(str(CAN / 'rgb.mp4')); H, W = int(cap.get(4)), int(cap.get(3)); F = min(len(poses['HGA']['wooden_spoon']), int(cap.get(7)))
    fw = 720; ch = int(round(H * fw / W / 2)) * 2; zs = ch; ph = 130; tw = fw + zs
    proc = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{tw}x{ch + 30 + ph}', '-r', str(fps), '-i', '-',
                             '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '26', '-movflags', '+faststart', pre + '.mp4'], stdin=subprocess.PIPE)

    def proj(T, P):
        q = (Kc @ (T[:3, :3] @ np.atleast_2d(P).T + T[:3, 3:])).T; return q[:, :2] / q[:, 2:]

    def sil(T, P):
        uv = np.round(proj(T, P)).astype(int); m = np.zeros((H, W), np.uint8)
        ok = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H); m[uv[ok, 1], uv[ok, 0]] = 1
        return cv2.morphologyEx(cv2.dilate(m, np.ones((3, 3), np.uint8)), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)) > 0
    per, sheet = [], {}
    L = np.linalg.norm(bowl - tail)
    for f in range(F):
        img = cap.read()[1]; im = img.copy()
        mk = cv2.imread(str(CAN / 'objects/wooden_spoon/masks/1' / f'{f:06d}.png'), 0)
        if mk is not None:
            cs_, _ = cv2.findContours((mk > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(im, cs_, -1, (255, 255, 255), 1, cv2.LINE_AA)
        row = dict(f=f)
        for tag in SUBS:
            row[f'deg_{tag}'] = round(float(out[tag][2][f, k]), 1)
            row[f'tilt_{tag}'], row[f'roll_{tag}'] = [round(x, 1) for x in turn(out[tag][0]['wooden_spoon'][f], out[tag][1]['wooden_spoon'][f])]
        for tag in ('v29', 'v31a', 'HGA'):
            for n in names:
                if n == 'wooden_spoon':
                    continue
                cs_, _ = cv2.findContours(sil(poses[tag][n][f], pts[n]).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(im, cs_, -1, COL[tag], 1, cv2.LINE_AA)
            T = poses[tag]['wooden_spoon'][f]
            cs_, _ = cv2.findContours(sil(T, pts['wooden_spoon']).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(im, cs_, -1, COL[tag], 2, cv2.LINE_AA)
            b, t_, cb, tk = proj(T, np.stack([bowl, tail, bowl - lax * 0.2 * L, bowl - lax * 0.2 * L + cross * 0.06]))
            cv2.line(im, tuple(np.int32(t_)), tuple(np.int32(b)), COL[tag], 1, cv2.LINE_AA); cv2.circle(im, tuple(np.int32(b)), 6, COL[tag], -1, cv2.LINE_AA)
            cv2.line(im, tuple(np.int32(cb)), tuple(np.int32(tk)), COL[tag], 3, cv2.LINE_AA)
        per.append(row)
        full = cv2.resize(im, (fw, ch), interpolation=cv2.INTER_AREA)
        cx, cy = proj(poses['HGA']['wooden_spoon'][f], (bowl + tail) / 2)[0]; hs = 190
        x0, y0 = int(np.clip(cx - hs, 0, W - 2 * hs)), int(np.clip(cy - hs, 0, H - 2 * hs)); zoom = cv2.resize(im[y0:y0 + 2 * hs, x0:x0 + 2 * hs], (zs, zs), interpolation=cv2.INTER_CUBIC)
        if f in SHEET:
            sheet[f] = cv2.resize(zoom, (300, 300), interpolation=cv2.INTER_AREA)
            cv2.putText(sheet[f], f'f{f}  v31a {row["deg_v31a"]:.0f}', (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
        top = np.full((30, tw, 3), 24, np.uint8); x = 8
        for tag, lab in (('HGA', 'HGA recon (target)'), ('v31a', f'v31a robot {row["deg_v31a"]:.0f} deg'), ('v29', f'v29 robot {row["deg_v29"]:.0f} deg')):
            cv2.putText(top, lab, (x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL[tag], 1, cv2.LINE_AA); x += 260
        cv2.putText(top, f'f{f}', (tw - 60, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
        plot = np.full((ph, tw, 3), 18, np.uint8); xs = lambda t: int(40 + t / max(F - 1, 1) * (tw - 50)); ys = lambda v: int(ph - 8 - min(v, 180.0) / 180.0 * (ph - 22))
        for v_, col in ((15, (60, 200, 220)), (30, (80, 80, 220))):
            cv2.line(plot, (40, ys(v_)), (tw - 10, ys(v_)), col, 1)
        for tag in ('v29', 'v31a'):
            p_ = np.array([[xs(r['f']), ys(r[f'deg_{tag}'])] for r in per], np.int32)
            if len(p_) > 1:
                cv2.polylines(plot, [p_], False, COL[tag], 1, cv2.LINE_AA)
        cv2.line(plot, (xs(f), 0), (xs(f), ph), (200, 200, 200), 1)
        cv2.putText(plot, 'spoon orientation error vs HGA (0-180 deg; yellow 15, red 30)', (44, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
        proc.stdin.write(np.vstack([top, np.hstack([full, zoom]), plot]).tobytes())
        if f == 170:
            cv2.imwrite(pre + '_f170.jpg', np.vstack([top, np.hstack([full, zoom]), plot]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    proc.stdin.close(); proc.wait()
    tiles = [sheet[f] for f in SHEET if f in sheet]; rows_ = [np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]
    cv2.imwrite(pre + '_sheet.jpg', np.vstack(rows_), [cv2.IMWRITE_JPEG_QUALITY, 85])
    Path(pre + '.json').write_text(json.dumps(dict(per_frame=per)))
    for a, b in ((158, 162), (163, 177), (178, 182)):
        sel = [r for r in per if a <= r['f'] <= b]
        print(f'f{a}-{b}: ' + '  '.join(f'{t} err {np.mean([r[f"deg_{t}"] for r in sel]):.0f} tilt {np.mean([r[f"tilt_{t}"] for r in sel]):.0f} roll {np.median([r[f"roll_{t}"] for r in sel]):.0f}' for t in SUBS))


if __name__ == '__main__':
    main()
