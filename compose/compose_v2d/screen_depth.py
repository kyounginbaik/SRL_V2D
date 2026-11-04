"""GT-free depth-consistency screen for tracked sequences.
The tracked metric mesh is rendered (sample z-buffer, 1/4 res) into the rect-left camera of every 2nd frame and compared
with the observed stereo depth inside the object mask (hands and the other objects removed). A flipped or misplaced pose
claims a surface where the depth sees none: err = z_obs - z_render >> 0 (e.g. a lid-on pot anchored upside down shows the
closed bottom at rim height while the depth sees the interior). Reported per run: median coverage of the mask by the mesh,
median err, mean fraction of pixels with err > 2 cm, fraction of 'bad' frames (deep fraction > 0.3 or coverage < 0.5).
usage: screen_depth.py [--episodes ...] [--tags _v3 _v9] [--variants icp_track_6dof compose_meshicp_smooth] [--out outputs/screen_depth.json]
"""
import argparse, glob, json, os, sys
import cv2, numpy as np, trimesh
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, work_dir, mono_masks_dir, load_depth, split_of
from v2d_io import episode_objects, PUBLIC_EPISODES, EVAL_EPISODES


def zbuffer(Pc, K, H, W, s):
    u = np.round(Pc[:, 0] / Pc[:, 2] * K[0, 0] / s + K[0, 2] / s).astype(int); v = np.round(Pc[:, 1] / Pc[:, 2] * K[1, 1] / s + K[1, 2] / s).astype(int)
    ok = (Pc[:, 2] > 0.05) & (u >= 0) & (u < W // s) & (v >= 0) & (v < H // s)
    zb = np.full((H // s) * (W // s), np.inf); idx = v[ok] * (W // s) + u[ok]; z = Pc[ok, 2]
    order = np.argsort(-z); zb[idx[order]] = z[order]                      # far first, near last -> min depth per cell
    return zb.reshape(H // s, W // s)


def screen(ep, obj, tag, variant, depth_all, dframes, K, others, s=4, step=2):
    td = work_dir(ep, obj) / f'track{tag}'
    z = np.load(td / 'poses.npz')
    if variant not in z.files: return None
    T = z[variant]; E = z['E']; md = mono_masks_dir(ep, obj); n = len(T)
    m_tri = trimesh.load(str(td / 'mesh_metric.obj'), force='mesh'); np.random.seed(0); g, _ = trimesh.sample.sample_surface(m_tri, 40000)
    res = []
    for f in range(0, n, step):
        if f not in dframes: continue
        d = depth_all[dframes.index(f)]; H, W = d.shape
        m = cv2.imread(str(md / f'obj_{f:05d}.png'), 0); h = cv2.imread(str(md / f'hand_{f:05d}.png'), 0)
        if m is None: continue
        m = m > 0
        if h is not None: m &= ~(cv2.dilate((h > 0).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0)
        for o in others:
            mo = cv2.imread(str(mono_masks_dir(ep, o) / f'obj_{f:05d}.png'), 0)
            if mo is not None:
                keep = m & ~(cv2.dilate((mo > 0).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0)
                if keep.sum() >= max(300, 0.15 * m.sum()): m = keep                 # unless the other mask swallows this object (pot mask on the lid)
        m &= d > 0
        if m.sum() < 300: continue
        Pw = g @ T[f, :3, :3].T + T[f, :3, 3]; Ei = np.linalg.inv(E[f]); Pc = Pw @ Ei[:3, :3].T + Ei[:3, 3]
        zb = zbuffer(Pc, K, H, W, s)
        ms = cv2.resize(m.astype(np.uint8), (W // s, H // s), interpolation=cv2.INTER_NEAREST) > 0
        ds = cv2.resize(d, (W // s, H // s), interpolation=cv2.INTER_NEAREST).astype(np.float64) / 1000
        both = ms & np.isfinite(zb) & (ds > 0)
        cover = both.sum() / max(ms.sum(), 1); err = (ds - zb)[both]
        # silhouette spill: rendered mesh cells outside the (dilated) object mask that are not hidden by a hand / other object
        # -> a pot handle that slipped around the symmetry axis renders onto the table (invisible to the depth test above)
        m_all = cv2.imread(str(md / f'obj_{f:05d}.png'), 0) > 0
        if h is not None: m_all |= cv2.dilate((h > 0).astype(np.uint8), np.ones((25, 25), np.uint8)) > 0
        for o in others:
            mo = cv2.imread(str(mono_masks_dir(ep, o) / f'obj_{f:05d}.png'), 0)
            if mo is not None: m_all |= mo > 0
        ms_all = cv2.dilate(cv2.resize(m_all.astype(np.uint8), (W // s, H // s), interpolation=cv2.INTER_NEAREST), np.ones((5, 5), np.uint8)) > 0
        rendered = np.isfinite(zb)
        spill = float(((rendered & ~ms_all).sum()) / max(rendered.sum(), 1))
        if len(err) < 20: res.append((f, cover, np.nan, np.nan, spill)); continue
        res.append((f, cover, float(np.median(err)), float(np.mean(err > 0.02)), spill))
    if not res: return None
    r = np.array(res, np.float64)
    bad = (np.nan_to_num(r[:, 3], nan=1.0) > 0.3) | (r[:, 1] < 0.5)
    return dict(frames=int(len(r)), cover_med=float(np.nanmedian(r[:, 1])), err_med_mm=float(np.nanmedian(r[:, 2]) * 1000), frac_deep=float(np.nanmean(r[:, 3])),
                bad_frames=float(bad.mean()), bad_first=int(r[np.argmax(bad), 0]) if bad.any() else -1, bad_last_q=float(bad[len(bad) // 2:].mean()),
                spill_med=float(np.median(r[:, 4])), spill_p90=float(np.percentile(r[:, 4], 90)), spill_frames=float(np.mean(r[:, 4] > 0.15)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', nargs='*', type=int, default=None); ap.add_argument('--tags', nargs='*', default=None)
    ap.add_argument('--variants', nargs='*', default=['icp_track_6dof', 'compose_meshicp_smooth']); ap.add_argument('--out', default=str(ROOT / 'outputs/screen_depth.json'))
    a = ap.parse_args()
    eps = a.episodes or (PUBLIC_EPISODES + EVAL_EPISODES)
    out = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for ep in eps:
        depth_all, dframes, dmeta = load_depth(ep); K = np.array(dmeta['K_rect'], np.float64); objs = episode_objects(ep)
        for obj in objs:
            others = [o for o in objs if o != obj]
            for td in sorted(work_dir(ep, obj).glob('track_*')):
                tag = td.name[5:]
                if a.tags and tag not in a.tags: continue
                if not (td / 'poses.npz').exists() or not (td / 'mesh_metric.obj').exists(): continue
                res = json.load(open(td / 'results.json')).get('results', {}) if (td / 'results.json').exists() else {}
                for v in a.variants:
                    key = f'{ep}:{obj}:{tag}:{v}'
                    if key in out: continue
                    r = screen(ep, obj, tag, v, depth_all, dframes, K, others)
                    if r is None: continue
                    r['auc'] = res.get(v, {}).get('d0', {}).get('auc'); out[key] = r
                    print(f"{key:48s} cover {r['cover_med']:.2f} err {r['err_med_mm']:6.1f} mm deep {r['frac_deep']:.2f} bad {r['bad_frames']:.2f} (first {r['bad_first']:4d}, 2nd half {r['bad_last_q']:.2f}) spill {r['spill_med']:.2f}/{r['spill_p90']:.2f} ({r['spill_frames']:.2f})" + (f" | AUC {r['auc']:.3f}" if r['auc'] is not None else ''), flush=True)
                    json.dump(out, open(a.out, 'w'), indent=1)
    print('done')


if __name__ == '__main__':
    main()
