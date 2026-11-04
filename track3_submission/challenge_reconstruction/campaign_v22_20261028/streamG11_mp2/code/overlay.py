"""GT-free overlay of a bundle episode: projected mesh surface points over RGB frames. usage: overlay.py EPROOT BUNDLE_EP_DIR OUT.jpg f1,f2,... [crop x0,y0,x1,y1]"""
import sys, json, numpy as np, cv2, trimesh
root, bd, out, fr = sys.argv[1], sys.argv[2], sys.argv[3], [int(x) for x in sys.argv[4].split(',')]
crop = [int(x) for x in sys.argv[5].split(',')] if len(sys.argv) > 5 else None
d = np.load(f'{bd}/object_poses.npz'); names = [str(n) for n in d['names']]; mj = json.load(open(f'{bd}/meshes.json'))
intr = json.load(open(f'{root}/intrinsics.json')); K = np.array([[intr['fx'], 0, intr['cx']], [0, intr['fy'], intr['cy']], [0, 0, 1.]])
cols = [(0, 255, 255), (255, 0, 255), (0, 255, 0)]
P = []
for n in names:
    f = mj[n]; f = f['file'] if isinstance(f, dict) else f
    m = trimesh.load(f if f.startswith('/') else f'{bd}/{f}', force='mesh', process=False); p, fi = trimesh.sample.sample_surface(m, 20000, seed=0); P.append((p, m.face_normals[fi]))
cap = cv2.VideoCapture(f'{root}/rgb.mp4'); ims = {}; i = 0
while True:
    ok, im = cap.read()
    if not ok: break
    if i in fr: ims[i] = im
    i += 1
tiles = []
for f in fr:
    im = ims[f].copy(); cam = np.linalg.inv(d['world_from_rgb'][f])
    for s, (p, nn) in enumerate(P):
        T = cam @ d['world_from_object'][f, s]; c = p @ T[:3, :3].T + T[:3, 3]; fac = np.einsum('ij,ij->i', nn @ T[:3, :3].T, c) < 0
        c = c[fac]; uv = (c[:, :2] / c[:, 2:]) @ K[:2, :2].T + K[:2, 2]
        for u, v in uv[::4].astype(int):
            if 0 <= u < im.shape[1] and 0 <= v < im.shape[0]: im[v, u] = cols[s]
    cv2.putText(im, str(f), (20, 50), 0, 1.5, (0, 0, 255), 3)
    if crop: im = im[crop[1]:crop[3], crop[0]:crop[2]]
    tiles.append(im)
rows = [np.concatenate(tiles[i:i + 3], 1) for i in range(0, len(tiles), 3)]
w = max(r.shape[1] for r in rows); rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows]
cv2.imwrite(out, cv2.resize(np.concatenate(rows, 0), None, fx=0.6, fy=0.6))
