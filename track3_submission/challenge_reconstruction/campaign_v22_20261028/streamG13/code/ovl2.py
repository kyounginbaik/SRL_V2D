"""Contour overlay: SAM2 mask contour (white) and the rendered silhouette contour of each hypothesis (green, red, magenta, cyan).
usage: ovl2.py EPROOT OUT.jpg f1,f2,.. OBJ x0,y0,x1,y1|full HYP1.npz HYP2.npz ..."""
import sys, json, numpy as np, cv2, trimesh
root, out, fr, obj, crop = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3].split(',')], sys.argv[4], sys.argv[5]
cols = [(0, 255, 0), (0, 0, 255), (255, 0, 255), (255, 255, 0)]
hy = [np.load(h, allow_pickle=True) for h in sys.argv[6:]]
intr = json.load(open(f'{root}/intrinsics.json')); K = np.array([[intr['fx'], 0, intr['cx']], [0, intr['fy'], intr['cy']], [0, 0, 1.]]); W, Hh = intr['width'], intr['height']
cache = {}
def pts(m):
    if m not in cache:
        mm = trimesh.load(m, force='mesh', process=False); p, fi = trimesh.sample.sample_surface(mm, 150000, seed=1); cache[m] = p
    return cache[m]
cap = cv2.VideoCapture(f'{root}/rgb.mp4'); ims = {}; i = 0
while True:
    ok, im = cap.read()
    if not ok: break
    if i in fr: ims[i] = im
    i += 1
tiles = []
for f in fr:
    im = ims[f].copy()
    mk = cv2.imread(f'{root}/objects/{obj}/masks/1/{f:06d}.png', 0)
    if mk is not None:
        cs, _ = cv2.findContours((mk > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(im, cs, -1, (255, 128, 0), 3)
    for h, H in enumerate(hy):
        names = [str(x) for x in H['names']]; s = names.index(obj)
        p = pts(str(H['meshes'][s])); T = H['camera_from_object'][f, s]; c = p @ T[:3, :3].T + T[:3, 3]; c = c[c[:, 2] > .05]
        uv = np.rint((c[:, :2] / c[:, 2:]) @ K[:2, :2].T + K[:2, 2]).astype(int); ok = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < Hh)
        sil = np.zeros((Hh, W), np.uint8); sil[uv[ok, 1], uv[ok, 0]] = 1; sil = cv2.morphologyEx(sil, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        cs, _ = cv2.findContours(sil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(im, cs, -1, cols[h], 2)
    if crop != 'full':
        x0, y0, x1, y1 = [int(v) for v in crop.split(',')]; im = im[y0:y1, x0:x1]
    cv2.putText(im, str(f), (10, 35), 0, 1.0, (0, 0, 255), 2); tiles.append(im)
rows = [np.concatenate(tiles[i:i + 4], 1) for i in range(0, len(tiles), 4)]
w = max(r.shape[1] for r in rows); rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows]
cv2.imwrite(out, np.concatenate(rows, 0))
