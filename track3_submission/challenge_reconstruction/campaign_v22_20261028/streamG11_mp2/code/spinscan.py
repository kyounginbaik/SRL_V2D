"""GT-free spin scan for an order-2 knob/handle object (white_pot_lid): apply a body-frame spin delta about the mesh's
thin (knob) axis through the centroid to the whole track and measure image evidence per frame:
  ev   = fuse.evidence cost (mask IoU + stereo depth), full resolution
  edge = handle edge chamfer: boundary of the projected handle region (mesh points > H above the disc) vs Canny edges
         of the RGB inside the dilated object mask (truncated distance, px)
  hd   = stereo depth residual on the projected handle pixels only (mm, median |z_render - z_stereo|)
No reference data is read.
usage: spinscan.py EPISODE_ROOT POSES_DIR OBJECT OUT.npz [--step 5] [--every 3]"""
import sys, json, argparse
from pathlib import Path
import numpy as np, cv2, trimesh
from scipy.spatial.transform import Rotation
sys.path.insert(0, '/data1/SRL_V2D/tools/reconstruction_audit')
from fuse import evidence

ap = argparse.ArgumentParser(); ap.add_argument('root', type=Path); ap.add_argument('poses', type=Path); ap.add_argument('obj'); ap.add_argument('out')
ap.add_argument('--step', type=float, default=5); ap.add_argument('--every', type=int, default=3); ap.add_argument('--height', type=float, default=0.012)
a = ap.parse_args(); root = a.root
d = np.load(a.poses / 'object_poses.npz'); names = [str(n) for n in d['names']]; s = names.index(a.obj)
mj = json.loads((a.poses / 'meshes.json').read_text()) if (a.poses / 'meshes.json').exists() else {}
mp = mj.get(a.obj, str(root / 'objects' / a.obj / 'mesh.obj'))
if isinstance(mp, dict): mp = str(a.poses / mp['file'])
mesh = trimesh.load(mp, force='mesh', process=False)
v = np.asarray(mesh.vertices); c = v.mean(0); w, U = np.linalg.eigh(np.cov((v - c).T)); ax = U[:, 0]
h = (v - c) @ ax; skew = np.mean(h ** 3) / np.std(h) ** 3; ax = ax * np.sign(skew)
pts, fi = trimesh.sample.sample_surface(mesh, 80000, seed=1); nrm = np.asarray(mesh.face_normals)[fi]
hh = (pts - c) @ ax; disc_top = np.percentile(hh, 60)
handle = hh > disc_top + a.height
print('mesh', mp, 'skew', round(float(skew), 2), 'handle pts', int(handle.sum()), flush=True)
intr = json.loads((root / 'intrinsics.json').read_text()); K = np.array([[intr['fx'], 0, intr['cx']], [0, intr['fy'], intr['cy']], [0, 0, 1.]])
W = d['world_from_object'][:, s]; cam = np.linalg.inv(d['world_from_rgb']); obs = d['object_observed'][:, s] if 'object_observed' in d else np.ones(len(W), bool)
deltas = np.arange(-90, 90, a.step)
Ts = []
for dl in deltas:
    T = np.eye(4); T[:3, :3] = Rotation.from_rotvec(np.deg2rad(dl) * ax).as_matrix(); T[:3, 3] = c - T[:3, :3] @ c; Ts.append(T)
cap = cv2.VideoCapture(str(root / 'rgb.mp4')); frames_rgb = []
ok = True
while ok:
    ok, im = cap.read()
    if ok: frames_rgb.append(im)
frames = [f for f in np.flatnonzero(obs) if f % a.every == 0 and f < len(frames_rgb)]
EV = np.full((len(frames), len(deltas)), np.nan); ED = EV.copy(); HD = EV.copy(); IOU = EV.copy()
for fi_, f in enumerate(frames):
    mfile = root / 'objects' / a.obj / 'masks/1' / f'{f:06d}.png'
    if not mfile.exists(): continue
    mask = cv2.imread(str(mfile), 0) > 0
    if mask.sum() < 200: continue
    exclude = np.zeros_like(mask)
    for lab in '23':
        p = root / 'objects' / a.obj / 'masks' / lab / f'{f:06d}.png'
        if p.exists(): exclude |= cv2.imread(str(p), 0) > 0
    exclude = (cv2.dilate(exclude.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) & ~mask
    enc = cv2.imread(str(root / 'depth' / f'{f:06d}.png'), -1)
    depth = 65535 / np.maximum(enc.astype(float), 1) - 1; depth[(enc == 0) | (enc >= 65535)] = np.nan
    gray = cv2.cvtColor(frames_rgb[f], cv2.COLOR_BGR2GRAY); gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 30, 90) > 0
    region = cv2.dilate(mask.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    edges &= region
    dt = cv2.distanceTransform((~edges).astype(np.uint8), cv2.DIST_L2, 3)
    for k, T in enumerate(Ts):
        pose = cam[f] @ W[f] @ T
        cost, info = evidence(mesh, pts, nrm, pose, K, mask, exclude, depth)
        EV[fi_, k] = cost; IOU[fi_, k] = info['visible_iou']
        cl = pts[handle] @ pose[:3, :3].T + pose[:3, 3]; nn = nrm[handle] @ pose[:3, :3].T
        fac = (np.einsum('ij,ij->i', nn, cl) < 0) & (cl[:, 2] > .05)
        cl = cl[fac]
        if len(cl) < 30: continue
        uv = cl[:, :2] / cl[:, 2:] @ K[:2, :2].T + K[:2, 2]; uv = np.rint(uv).astype(int)
        ins = (uv[:, 0] >= 0) & (uv[:, 0] < mask.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < mask.shape[0]); uv, cl = uv[ins], cl[ins]
        if len(uv) < 30: continue
        hm = np.zeros(mask.shape, np.uint8); hm[uv[:, 1], uv[:, 0]] = 1; hm = cv2.morphologyEx(hm, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        bd = (hm > 0) & ~(cv2.erode(hm, np.ones((3, 3), np.uint8)) > 0)
        bd &= ~exclude
        if bd.sum() < 20: continue
        ED[fi_, k] = float(np.mean(np.minimum(dt[bd], 15)))
        z = np.full(mask.shape, np.inf); np.minimum.at(z, (uv[:, 1], uv[:, 0]), cl[:, 2])
        use = np.isfinite(z) & np.isfinite(depth) & mask & ~exclude
        if use.sum() >= 12: HD[fi_, k] = float(np.median(np.abs(z[use] - depth[use])) * 1000)
    print(f, 'ev', deltas[np.nanargmin(EV[fi_])] if np.isfinite(EV[fi_]).any() else None, 'edge', deltas[np.nanargmin(ED[fi_])] if np.isfinite(ED[fi_]).any() else None,
          'hd', deltas[np.nanargmin(HD[fi_])] if np.isfinite(HD[fi_]).any() else None, flush=True)
np.savez_compressed(a.out, deltas=deltas, frames=np.array(frames), ev=EV, edge=ED, hd=HD, iou=IOU, axis=ax, centroid=c)
for nm, X in [('ev', EV), ('edge', ED), ('hd', HD)]:
    ok = np.isfinite(X).all(1)
    if ok.sum():
        m = np.nanmean(X[ok] - np.nanmin(X[ok], 1, keepdims=True), 0)
        print('POOLED', nm, 'frames', int(ok.sum()), 'best', float(deltas[np.argmin(m)]), 'curve', dict(zip(deltas[::3].tolist(), m[::3].round(2).tolist())), flush=True)
