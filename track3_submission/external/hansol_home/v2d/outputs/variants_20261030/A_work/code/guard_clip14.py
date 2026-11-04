"""Variant A on the re-timed base (selected20_final_v14_g17 / public g15_camrt): the same smoother + image guard as
guard_clip.py, with clock-consistent evidence (GT-free).

G15 resampled every pose of v14_g17 / g15_camrt at t + delta (RGB ego_cam_a clock -> stereo ego_cam_b/c clock,
streamG15/code/retime.py; delta per clip from streamG15/out/sync_delta_t.json). The SAM2 masks and the spoon ball
detections are RGB-clock, the stereo depth is b/c-clock. So the guard evaluates
  - silhouette IoU and ball reprojection with the track shifted back by -delta (team retime.shift), against RGB frame g;
  - median |mesh - stereo depth| with the track as stored (index g = depth frame g).
Base and smoothed tracks go through the same evaluation. Smoothing acts on world_from_object; camera_from_object gets
the same body-frame change (cfo_new = cfo_old @ inv(wfo_old) @ wfo_new), so G15's own wfo/cfo interpolation residual
(<= 2e-4) is untouched. Output npz are savez_compressed like v14_g17.

usage: guard_clip14.py public|eval EPISODE_DIR PARAMS_JSON [--check] [--src-export NAME]
  public: EPISODE_DIR = A_work/pub/<split>/episode_X (input exports/g15_camrt, output exports/A14_<tag>)
  eval:   EPISODE_DIR = v14_g17/episode_X, or the same episode of another v14-clock bundle (output A_work/eval14/<tag>/episode_X)
  --check: only print base evidence with/without the clock shift (no output)
  --src-export NAME: public input export instead of g15_camrt (must be on the same G15 clock, e.g. v14_B2)"""
import sys, json, shutil, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG15/code')
import slib
from guard_clip import Evidence, Balls, W, CAN
from retime import shift

DELTA = json.loads(Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG15/out/sync_delta_t.json').read_text())


def deltas(key, F):
    d = DELTA[key]
    return np.asarray(d, float)[:F] if np.ndim(d) else np.full(F, float(d))


def evaluate(ev, n, cam, d, frames_rgb, frames_idx, balls):
    cam_rgb = shift(cam, -d)
    iou, _ = ev.eval(n, cam_rgb, frames_rgb)
    _, z = ev.eval(n, cam, frames_idx)
    b = balls.eval(cam_rgb, frames_rgb) if balls else np.full(len(cam), np.nan)
    return iou, z, b, cam_rgb


def guarded(T, cam, info, ev, name, balls, d, P):
    sp = {k: v for k, v in P.items() if k not in ('d_iou', 'd_z', 'd_ball', 'rounds')}
    d_iou, d_z, d_ball = P.get('d_iou', .05), P.get('d_z', 3.0), P.get('d_ball', 3.0)
    F = len(T); allf = range(F)
    ib, zb, bb, cam_b_rgb = evaluate(ev, name, cam, d, allf, allf, balls)
    tq = np.clip(np.arange(F) - d, 0, F - 1); lo = np.floor(tq).astype(int); hi = np.minimum(lo + 1, F - 1)
    wm = np.ones(F); hist = []
    for rnd in range(P.get('rounds', 8)):
        Ts, extra = slib.smooth_track(T, info, wmult=wm, **sp)
        cam_s = cam @ np.linalg.inv(T) @ Ts
        cam_s_rgb = shift(cam_s, -d)
        ch_idx = np.where(np.any(np.abs(cam_s - cam) > 1e-12, axis=(1, 2)))[0]
        ch_rgb = np.where(np.any(np.abs(cam_s_rgb - cam_b_rgb) > 1e-12, axis=(1, 2)))[0]
        i_s, _ = ev.eval(name, cam_s_rgb, ch_rgb); _, z_s = ev.eval(name, cam_s, ch_idx)
        b_s = balls.eval(cam_s_rgb, ch_rgb) if balls else np.full(F, np.nan)
        with np.errstate(invalid='ignore'):
            bad_rgb = np.zeros(F, bool); bad_rgb[ch_rgb] = (i_s[ch_rgb] < ib[ch_rgb] - d_iou) | (b_s[ch_rgb] > bb[ch_rgb] + d_ball)
            bad_idx = np.zeros(F, bool); bad_idx[ch_idx] = np.abs(z_s[ch_idx]) > np.abs(zb[ch_idx]) + d_z
        bad = bad_idx.copy(); bad[lo[bad_rgb]] = True; bad[hi[bad_rgb]] = True
        hist.append(int(bad_rgb.sum() + bad_idx.sum()))
        if not bad.any():
            break
        wm[bad] *= 10.0
    i_s, z_s, b_s, _ = evaluate(ev, name, cam_s, d, allf, allf, balls)
    dr = slib.rel_angle_deg(Ts[:, :3, :3], T[:, :3, :3], info)
    dp = np.linalg.norm(slib.centers(Ts, info) - slib.centers(T, info), axis=-1) * 100
    nm = lambda x: float(np.nanmean(x)) if np.isfinite(x).any() else None
    md = lambda x: float(np.nanmedian(x)) if np.isfinite(x).any() else None
    pf = lambda x: [None if not np.isfinite(v) else round(float(v), 3) for v in x]
    return Ts, cam_s, dict(extra, rounds=len(hist), bad_per_round=hist, guarded_frames=int((wm > 1).sum()), final_bad=hist[-1],
                           iou_mean=[nm(ib), nm(i_s)], absz_mean=[nm(np.abs(zb)), nm(np.abs(z_s))], ball_median=[md(bb), md(b_s)],
                           iou_drop_gt0p1=int(np.nansum(i_s - ib < -.1)), change_rot_mean=float(dr.mean()), change_pos_mean=float(dp.mean()),
                           change_rot_max=float(dr.max()), change_pos_max=float(dp.max()),
                           per_frame=dict(iou_rgbclock=[pf(ib), pf(i_s)], absz_stereoclock=[pf(np.abs(zb)), pf(np.abs(z_s))], ball_rgbclock=[pf(bb), pf(b_s)]))


def main():
    part, ep, params = sys.argv[1], Path(sys.argv[2]), json.loads(Path(sys.argv[3]).read_text())
    check = '--check' in sys.argv
    src_export = sys.argv[sys.argv.index('--src-export') + 1] if '--src-export' in sys.argv else 'g15_camrt'
    t0 = time.time(); e = int(ep.name[-6:])
    if part == 'public':
        src = ep / 'exports' / src_export; root = CAN / ep.parent.name / ep.name
        meshes = json.loads((src / 'meshes.json').read_text()); key = f'public_{e}'
        outs = {tag: ep / 'exports' / f'A14_{tag}' for tag in params}
    else:
        src = ep; root = CAN / 'evaluation' / ep.name; key = f'evaluation_{e}'
        meshes = {n: str(ep / v['file']) for n, v in json.loads((ep / 'meshes.json').read_text()).items()}
        outs = {tag: W / 'eval14' / tag / ep.name for tag in params}
    todo = {t: o for t, o in outs.items() if not (o / 'object_poses.npz').exists()}
    if not todo and not check:
        return
    dz = dict(np.load(src / 'object_poses.npz'))
    names = [str(n) for n in dz['names']]; F = len(dz['world_from_object'])
    d = deltas(key, F)
    ev = Evidence(root, names, meshes, F)
    balls = Balls(root, F) if 'wooden_spoon' in names else None
    infos = {n: slib.mesh_info(n, meshes[n]) for n in names}
    if check:
        for k, n in enumerate(names):
            cam = dz['camera_from_object'][:, k]; b = balls if n == 'wooden_spoon' else None
            i0, z0 = ev.eval(n, cam, range(F)); i1, z1, b1, cam_rgb = evaluate(ev, n, cam, d, range(F), range(F), b)
            _, zm = ev.eval(n, shift(cam, -d), range(F))
            b0 = b.eval(cam, range(F)) if b else np.full(F, np.nan)
            m = lambda x: round(float(np.nanmean(x)), 4) if np.isfinite(x).any() else None
            print(f'{part} {ep.name} {n:18s} delta {sorted(set(d.tolist()))} | IoU as stored {m(i0)} vs shifted -delta {m(i1)} | '
                  f'|z| as stored {m(np.abs(z0))} vs shifted {m(np.abs(zm))} | ball as stored {m(b0)} vs shifted {m(b1)}', flush=True)
        return
    for tag, out in todo.items():
        wfo = dz['world_from_object'].copy(); cfo = dz['camera_from_object'].copy(); rep = {}
        for k, n in enumerate(names):
            T, C, r = guarded(dz['world_from_object'][:, k], dz['camera_from_object'][:, k], infos[n], ev, n, balls if n == 'wooden_spoon' else None, d, params[tag])
            r['before'] = slib.summarize(slib.motion_stats(dz['world_from_object'][:, k], infos[n])); r['after'] = slib.summarize(slib.motion_stats(T, infos[n]))
            ch = np.any(T != dz['world_from_object'][:, k], axis=(1, 2))
            wfo[:, k] = np.where(ch[:, None, None], T, dz['world_from_object'][:, k])
            cfo[:, k] = np.where(ch[:, None, None], C, dz['camera_from_object'][:, k])
            rep[n] = r
        o = dict(dz); o['world_from_object'] = wfo; o['camera_from_object'] = cfo
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / 'object_poses.npz', **o)
        if part == 'public':
            for f in ['meshes.json', 'selection.json', 'retime.json']:
                if (src / f).exists(): shutil.copy2(src / f, out / f)
        else:
            json.dump(meshes, open(out / 'meshes.json', 'w'), indent=1)
        (out / 'guard_report.json').write_text(json.dumps(dict(tag=tag, params=params[tag], source=str(src), delta_key=key,
                                                               delta_values=sorted(set(d.tolist())), objects=rep), indent=1))
        print(ep.name, tag, {n: (r['rounds'], r['guarded_frames'], r['final_bad'], [round(x, 3) if x is not None else None for x in r['iou_mean']], r['before']['rot_gt15'], r['after']['rot_gt15']) for n, r in rep.items()}, round(time.time() - t0, 1), 's', flush=True)


if __name__ == '__main__':
    main()
