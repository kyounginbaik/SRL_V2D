"""Variant A: image-guarded robust SE(3) smoothing of one clip, for several parameter sets (GT-free).

Rule (same for every clip and object): smooth (slib.smooth_track), then evaluate every changed frame with the team's
image evidence (fuse.evidence with qc.py's exact mask / exclusion / depth preparation): silhouette IoU against the SAM2
mask, median |mesh - stereo depth| and, for the spoon, gray-ball reprojection (spoon_roll_v8 model, our detections).
A frame whose smoothed pose is worse than the base pose (IoU drop > d_iou, |z| rise > d_z mm, ball rise > d_ball px)
gets its data weight multiplied by 10 and the track is re-smoothed, until no frame fails (max 8 rounds). So the
smoother only moves a frame where the image does not object: jitter, occluded / unobserved frames and spikes.

usage: guard_clip.py public|eval EPISODE_DIR PARAMS_JSON   (PARAMS_JSON = {tag: params})
  public: EPISODE_DIR = A_work/pub/<split>/episode_X (input exports/r8_base, output exports/A_<tag>)
  eval:   EPISODE_DIR = v8_g10b/episode_X (output A_work/eval/<tag>/episode_X)"""
import sys, json, shutil, time
from pathlib import Path
import cv2
import numpy as np
import trimesh
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, '/data1/SRL_V2D/tools/reconstruction_audit')
import slib
from fuse import evidence

W = Path('/home/hansol/v2d/outputs/variants_20261030/A_work')
CAN = Path('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/canonical')
cv2.setNumThreads(1)


class Evidence:
    """qc.py's per-frame evidence preparation, loaded once per clip."""

    def __init__(self, root, names, meshes, frames):
        intr = json.loads((root / 'intrinsics.json').read_text()); width, height = 254, 190
        self.k = np.array([[intr['fx'] * width / intr['width'], 0, intr['cx'] * width / intr['width']],
                           [0, intr['fy'] * height / intr['height'], intr['cy'] * height / intr['height']], [0, 0, 1.]])
        small = lambda path: (cv2.resize(value, (width, height), interpolation=cv2.INTER_NEAREST) > 0) if (value := cv2.imread(str(path), 0)) is not None else None
        self.names = names; self.frames = frames
        self.mesh = {}; self.data = {n: [None] * frames for n in names}
        for n in names:
            mesh = trimesh.load(meshes[n], force='mesh', process=False)
            points, faces = trimesh.sample.sample_surface(mesh, 6000, seed=260126)
            self.mesh[n] = (mesh, points, np.asarray(mesh.face_normals)[faces])
        for f in range(frames):
            encoded = cv2.resize(cv2.imread(str(root / 'depth' / f'{f:06d}.png'), -1), (width, height), interpolation=cv2.INTER_NEAREST)
            depth = 65535 / np.maximum(encoded.astype(float), 1) - 1; depth[(encoded == 0) | (encoded >= 65535)] = np.nan
            ms = {n: small(root / 'objects' / n / 'masks/1' / f'{f:06d}.png') for n in names}
            for n in names:
                mask = ms[n]
                others = [ms[o] for o in names if o != n]
                hands = [v for label in '23' if (v := small(root / 'objects' / n / 'masks' / label / f'{f:06d}.png')) is not None]
                exclude = np.zeros((height, width), np.uint8)
                for value in others + hands: exclude |= value.astype(np.uint8)
                exclude = cv2.dilate(exclude, np.ones((3, 3), np.uint8)) > 0
                keep = mask.copy()
                for value in others:
                    if value.sum() < mask.sum(): keep &= ~value
                exclude &= ~keep
                if mask.sum() < 20:
                    continue
                self.data[n][f] = (mask, exclude, depth)

    def eval(self, n, cam, frames):
        iou = np.full(self.frames, np.nan); z = np.full(self.frames, np.nan)
        mesh, points, normals = self.mesh[n]
        for f in frames:
            d = self.data[n][f]
            if d is None:
                continue
            _, info = evidence(mesh, points, normals, cam[f], self.k, d[0], d[1], d[2])
            iou[f] = info['visible_iou']
            if info['median_z_mm'] is not None: z[f] = info['median_z_mm']
        return iou, z


class Balls:
    def __init__(self, root, frames):
        self.ok = False
        bp = root / 'ball_detections_wooden_spoon.json'; sel = root / 'exports' / 'spoon_roll_v8' / 'selection.json'
        if bp.exists() and sel.exists():
            self.model = np.array(json.loads(sel.read_text())['objects']['wooden_spoon']['ball_roll']['model_body_m'])
            intr = json.loads((root / 'intrinsics.json').read_text())
            self.k = np.array([[intr['fx'], 0, intr['cx']], [0, intr['fy'], intr['cy']], [0, 0, 1.]])
            self.det = json.loads(bp.read_text())['frames']; self.ok = True

    def eval(self, cam, frames):
        per = np.full(len(cam), np.nan)
        if not self.ok: return per
        for f in frames:
            if f >= len(self.det) or not self.det[f]: continue
            pts = self.model @ cam[f, :3, :3].T + cam[f, :3, 3]
            uv = pts[:, :2] / pts[:, 2:] @ self.k[:2, :2].T + self.k[:2, 2]
            per[f] = np.median([np.min(np.linalg.norm(uv - b['uv'], axis=1)) for b in self.det[f]])
        return per


def guarded(T, info, inv_wfr, ev, name, balls, P):
    sp = {k: v for k, v in P.items() if k not in ('d_iou', 'd_z', 'd_ball', 'rounds')}
    d_iou, d_z, d_ball = P.get('d_iou', .05), P.get('d_z', 3.0), P.get('d_ball', 3.0)
    F = len(T); allf = range(F)
    cam_b = inv_wfr @ T
    ib, zb = ev.eval(name, cam_b, allf); bb = balls.eval(cam_b, allf) if balls else np.full(F, np.nan)
    wm = np.ones(F); hist = []
    for rnd in range(P.get('rounds', 8)):
        Ts, extra = slib.smooth_track(T, info, wmult=wm, **sp)
        dr = slib.rel_angle_deg(Ts[:, :3, :3], T[:, :3, :3], info)
        dp = np.linalg.norm(slib.centers(Ts, info) - slib.centers(T, info), axis=-1) * 100
        ch = np.where((dr > 1e-6) | (dp > 1e-6))[0]
        cam_s = inv_wfr @ Ts
        i_s, z_s = ev.eval(name, cam_s, ch); b_s = balls.eval(cam_s, ch) if balls else np.full(F, np.nan)
        with np.errstate(invalid='ignore'):
            bad = np.zeros(F, bool)
            bad[ch] = ((i_s[ch] < ib[ch] - d_iou) | (np.abs(z_s[ch]) > np.abs(zb[ch]) + d_z) | (b_s[ch] > bb[ch] + d_ball))
        hist.append(int(bad.sum()))
        if not bad.any():
            break
        wm[bad] *= 10.0
    i_s, z_s = ev.eval(name, cam_s, allf); b_s = balls.eval(cam_s, allf) if balls else np.full(F, np.nan)
    nm = lambda x: float(np.nanmean(x)) if np.isfinite(x).any() else None
    return Ts, dict(extra, rounds=len(hist), bad_per_round=hist, guarded_frames=int((wm > 1).sum()), final_bad=hist[-1],
                    iou_mean=[nm(ib), nm(i_s)], absz_mean=[nm(np.abs(zb)), nm(np.abs(z_s))], ball_median=[None if not np.isfinite(bb).any() else float(np.nanmedian(bb)), None if not np.isfinite(b_s).any() else float(np.nanmedian(b_s))],
                    iou_drop_gt0p1=int(np.nansum(i_s - ib < -.1)), change_rot_mean=float(dr.mean()), change_pos_mean=float(dp.mean()), change_rot_max=float(dr.max()), change_pos_max=float(dp.max()))


def main():
    part, ep, params = sys.argv[1], Path(sys.argv[2]), json.loads(Path(sys.argv[3]).read_text())
    t0 = time.time()
    if part == 'public':
        src = ep / 'exports' / 'r8_base'; root = CAN / ep.parent.name / ep.name
        meshes = json.loads((src / 'meshes.json').read_text())
        outs = {tag: ep / 'exports' / f'A_{tag}' for tag in params}
    else:
        src = ep; root = CAN / 'evaluation' / ep.name
        meshes = {n: str(ep / v['file']) for n, v in json.loads((ep / 'meshes.json').read_text()).items()}
        outs = {tag: W / 'eval' / tag / ep.name for tag in params}
    todo = {t: o for t, o in outs.items() if not (o / 'object_poses.npz').exists()}
    if not todo:
        return
    d = dict(np.load(src / 'object_poses.npz'))
    names = [str(n) for n in d['names']]; F = len(d['world_from_object'])
    ev = Evidence(root, names, meshes, F)
    balls = Balls(root, F) if 'wooden_spoon' in names else None
    inv_wfr = np.linalg.inv(d['world_from_rgb'])
    infos = {n: slib.mesh_info(n, meshes[n]) for n in names}
    print(ep.name, 'evidence loaded', round(time.time() - t0, 1), 's', flush=True)
    for tag, out in todo.items():
        wfo = d['world_from_object'].copy(); rep = {}
        for k, n in enumerate(names):
            T, r = guarded(d['world_from_object'][:, k], infos[n], inv_wfr, ev, n, balls if n == 'wooden_spoon' else None, params[tag])
            r['before'] = slib.summarize(slib.motion_stats(d['world_from_object'][:, k], infos[n])); r['after'] = slib.summarize(slib.motion_stats(T, infos[n]))
            wfo[:, k] = T; rep[n] = r
        o = dict(d)
        changed = np.any(wfo != d['world_from_object'], axis=(2, 3))
        if 'camera_from_object' in o:
            o['camera_from_object'] = np.where(changed[:, :, None, None], np.einsum('nij,nkjl->nkil', inv_wfr, wfo), d['camera_from_object'])
        o['world_from_object'] = wfo
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / 'object_poses.npz', **o)
        if part == 'public':
            for f in ['meshes.json', 'selection.json']:
                if (src / f).exists(): shutil.copy2(src / f, out / f)
        else:
            json.dump(meshes, open(out / 'meshes.json', 'w'), indent=1)
        (out / 'guard_report.json').write_text(json.dumps(dict(tag=tag, params=params[tag], source=str(src), objects=rep), indent=1))
        print(ep.name, tag, {n: (r['rounds'], r['guarded_frames'], r['final_bad'], [round(x, 3) if x is not None else None for x in r['iou_mean']], r['before']['rot_gt15'], r['after']['rot_gt15']) for n, r in rep.items()}, round(time.time() - t0, 1), 's', flush=True)


if __name__ == '__main__':
    main()
