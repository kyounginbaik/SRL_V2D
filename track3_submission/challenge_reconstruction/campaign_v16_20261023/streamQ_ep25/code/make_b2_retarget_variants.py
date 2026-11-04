"""Evidence helper: board-2 position retarget toward the coworker's NEW compose_eval track for f0-199 (rotation kept), on top of
the w3 (v3) corrected poses. Delta = NEW slab centre - OUR slab centre in our world (NEW via world_from_rgb @ A_T_rect @ inv(cam0_T_cam)
@ T_world_obj, h2h_other/code/common.py), gaussian-smoothed (sigma 2 frames), blended to 0 over f199-209.
variant 'full' = raw delta; variant 'rel' = delta minus its median over f200-239 (removes the constant mesh-frame offset between
the two board meshes, where both tracks agree on the motion). usage: make_b2_retarget_variants.py OUTDIR"""
import sys, numpy as np
from scipy.ndimage import gaussian_filter1d
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/h2h_other/code'); from common import load_ours, load_new
out = sys.argv[1]; o = load_ours(25); n = load_new(25, o['names'])
base = '/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamQ_ep25/world/w3/source_corrected/episode_000025/object_poses.npz'
z = dict(np.load(base)); W = z['world_from_object']; Wc = o['world_from_cam']
def slab(v):
    c = v.mean(0); w, U = np.linalg.eigh((v - c).T @ (v - c)); q = (v - c) @ U; h, e = np.histogram(q[:, 0], bins=80); pk = np.sort(e[np.argsort(h)[-2:]] + np.diff(e)[0] / 2)
    lo = np.percentile(q, .5, 0); hi = np.percentile(q, 99.5, 0); return c + U @ np.array([pk.mean(), (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2])
so = slab(np.asarray(o['meshes']['wooden_piece_2'].vertices)); sn = slab(np.asarray(n['meshes']['wooden_piece_2'].vertices))
D = np.array([(Wc[t] @ n['A_T_obj']['wooden_piece_2'][t])[:3, :3] @ sn + (Wc[t] @ n['A_T_obj']['wooden_piece_2'][t])[:3, 3] - (W[t, 1, :3, :3] @ so + W[t, 1, :3, 3]) for t in range(240)])
Ds = gaussian_filter1d(D, 2.0, axis=0, mode='nearest')
for name, off in (('full', np.zeros(3)), ('rel', np.median(D[200:240], 0))):
    T = W.copy()
    for t in range(0, 210):
        w = 1. if t <= 199 else 1 - (lambda u: u * u * (3 - 2 * u))((t - 199) / 10)
        T[t, 1, :3, 3] = W[t, 1, :3, 3] + w * (Ds[min(t, 199)] - off)
    zz = dict(z); zz['world_from_object'] = T; np.savez(f'{out}/variant_b2_{name}.npz', **zz)
    print(name, 'offset', (off * 100).round(2), 'shift cm mean/max f0-199', (np.linalg.norm(Ds[:200] - off, axis=1).mean() * 100).round(2), (np.linalg.norm(Ds[:200] - off, axis=1).max() * 100).round(2))
