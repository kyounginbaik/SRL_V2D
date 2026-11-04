"""Stream X8 (ep10): thinner SIM-ONLY collision proxies for blue_cup / beige_cup, derived from the radial profiles of OUR
reconstruction visual meshes (final_recon/episode_000010/assets, sha 57bb269a.. / 2c041e69..), with exactly the stream L
build_recon_world.py profile + sector construction (same samples, seed, bands, sectors, 1/cos push, base disk).
Modification (only the band radii change):
  blue_cup : inner radius R_in' = R_out - t_blue at every band edge (R_out = the stream L outer profile, unchanged).  This removes
             the 2.8 mm inward rim lip (band 15.7 cm) and thins the wall from the 1.5 mm clip to t_blue.
  beige_cup: whole wall shifted inward by d_beige: R_out' = R_out - d_beige, R_in' = R_in - d_beige (thickness unchanged).
Mass 0.1 kg kept, COM / inertia recomputed from the new parts exactly like stream L (shape-implied only).
No GT, no scan meshes."""
import hashlib, numpy as np, trimesh
from pathlib import Path
FINAL = Path('/data1/SRL_V2D/challenge_reconstruction/final_recon/episode_000010/assets')
EXPECTED = dict(beige_cup='2c041e69fbd3be79fa953609276ffc4a54a513ec2ca7e7f31020a78ab6871431',
                blue_cup='57bb269a06a44f9f56ea59c9eea8b0d704b20d5a8cfd8bbc89f73f5e2f2b1d8f')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def profile(n, NB=10):
    """verbatim stream L profile; returns dict with frame (c, a, e1, e2), edges, R_in (clipped), R_out, raw inner, base params"""
    vis = FINAL / f'{n}_visual.obj'; assert sha(vis) == EXPECTED[n], n
    m = trimesh.load(vis, force='mesh', process=False)
    P = trimesh.sample.sample_surface(m, 300000, seed=0)[0]; c = P.mean(0); w, Ue = np.linalg.eigh(np.cov((P - c).T)); a = Ue[:, -1]
    h = (P - c) @ a; rr = np.linalg.norm((P - c) - np.outer(h, a), axis=1)
    if rr[h > np.percentile(h, 85)].mean() < rr[h < np.percentile(h, 15)].mean(): a = -a; h = -h
    e1 = np.cross(a, [1, 0, 0]); e1 = e1 / np.linalg.norm(e1) if np.linalg.norm(e1) > .1 else np.cross(a, [0, 1, 0]) / np.linalg.norm(np.cross(a, [0, 1, 0])); e2 = np.cross(a, e1)
    ang = np.arctan2((P - c) @ e2, (P - c) @ e1); h0, h1 = h.min(), h.max()
    def radii(lo, hi):
        s = (h >= lo) & (h < hi); ins, outs = [], []
        for k in range(36):
            q = s & (ang >= -np.pi + k * 2 * np.pi / 36) & (ang < -np.pi + (k + 1) * 2 * np.pi / 36)
            if q.sum() > 5: ins.append(rr[q].min()); outs.append(rr[q].max())
        return (np.median(ins), np.median(outs)) if ins else (np.nan, np.nan)
    near = (rr < 0.015) & (h < h0 + 0.03); base_top = max(float(h[near].max()) + 0.001, h0 + 0.003) if near.any() else h0 + 0.004
    edges = np.linspace(base_top, h1, NB + 1); R_in, R_out = [], []
    for z in edges:
        ri, ro = radii(max(z - 0.004, base_top), z + 0.004); R_in.append(ri); R_out.append(ro)
    R_in, R_out = np.array(R_in), np.array(R_out); raw_in = R_in.copy(); thick = np.clip(R_out - R_in, 0.0015, 0.006); R_in = R_out - thick
    rb = radii(h0, base_top)[1]; rb = rb if np.isfinite(rb) else R_out[0]
    return dict(c=c, a=a, e1=e1, e2=e2, h0=h0, h1=h1, base_top=base_top, edges=edges, R_in=R_in, R_out=R_out, raw_in=raw_in, rb=rb)


def parts_from(pr, R_in, R_out, NA=24):
    c, a, e1, e2, edges = pr['c'], pr['a'], pr['e1'], pr['e2'], pr['edges']; NB = len(edges) - 1
    parts = []; da = 2 * np.pi / NA; push = 1 / np.cos(np.pi / NA)
    for b in range(NB):
        for k in range(NA):
            pts = []
            for z, ri, ro in ((edges[b], R_in[b], R_out[b]), (edges[b + 1], R_in[b + 1], R_out[b + 1])):
                for th in (k * da, (k + 1) * da):
                    d = np.cos(th) * e1 + np.sin(th) * e2
                    pts += [c + a * z + d * ri * push, c + a * z + d * ro * push]
            parts.append(trimesh.convex.convex_hull(np.array(pts)))
    base = []
    for th in np.linspace(0, 2 * np.pi, 33)[:-1]:
        d = np.cos(th) * e1 + np.sin(th) * e2; base += [c + a * pr['h0'] + d * pr['rb'], c + a * pr['base_top'] + d * R_out[0]]
    parts.append(trimesh.convex.convex_hull(np.array(base)))
    return parts


def inertia(parts, mass=0.1):
    vol = np.array([p.volume for p in parts]); rho = mass / vol.sum(); com = sum(p.volume * p.center_mass for p in parts) / vol.sum(); I = np.zeros((3, 3))
    for p in parts:
        m_i = rho * p.volume; dd = p.center_mass - com; I += p.moment_inertia / p.volume * m_i + m_i * (np.dot(dd, dd) * np.eye(3) - np.outer(dd, dd))
    return com, I


def modified(n, t_blue=None, d_beige=None):
    pr = profile(n); R_in, R_out = pr['R_in'].copy(), pr['R_out'].copy()
    if n == 'blue_cup' and t_blue is not None: R_in = R_out - t_blue
    if n == 'beige_cup' and d_beige is not None: R_in = R_in - d_beige; R_out = R_out - d_beige
    return pr, R_in, R_out
