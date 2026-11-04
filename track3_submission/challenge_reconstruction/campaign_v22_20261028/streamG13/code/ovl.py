"""Overlay several hypotheses (projected surface points, one colour each) + SAM2 mask contours (white) on RGB frames.
usage: ovl.py EPROOT OUT.jpg f1,f2,.. HYP1.npz[:bgr] HYP2.npz[:bgr] ..."""
import sys, json, numpy as np, cv2, trimesh
root, out, fr = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3].split(',')]
cols = [(0, 255, 0), (0, 0, 255), (255, 0, 255), (255, 255, 0)]
hy = [np.load(h, allow_pickle=True) for h in sys.argv[4:]]
intr = json.load(open(f'{root}/intrinsics.json')); K = np.array([[intr['fx'], 0, intr['cx']], [0, intr['fy'], intr['cy']], [0, 0, 1.]])
cache = {}
def pts(m):
    if m not in cache:
        mm = trimesh.load(m, force='mesh', process=False); p, fi = trimesh.sample.sample_surface(mm, 8000, seed=1); cache[m] = (p, mm.face_normals[fi])
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
    names = [str(n) for n in hy[0]['names']]
    for n in names:
        mk = cv2.imread(f'{root}/objects/{n}/masks/1/{f:06d}.png', 0)
        if mk is not None:
            cs, _ = cv2.findContours((mk > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(im, cs, -1, (255, 255, 255), 1)
    for h, H in enumerate(hy):
        for s, n in enumerate([str(x) for x in H['names']]):
            p, nn = pts(str(H['meshes'][s])); T = H['camera_from_object'][f, s]
            c = p @ T[:3, :3].T + T[:3, 3]; fac = np.einsum('ij,ij->i', nn @ T[:3, :3].T, c) < 0; c = c[fac]
            uv = (c[:, :2] / c[:, 2:]) @ K[:2, :2].T + K[:2, 2]
            for u, v in uv[::2].astype(int):
                if 0 <= u < im.shape[1] and 0 <= v < im.shape[0]: im[v, u] = cols[h]
    cv2.putText(im, str(f), (20, 50), 0, 1.5, (0, 0, 255), 3); tiles.append(im)
rows = [np.concatenate(tiles[i:i + 3], 1) for i in range(0, len(tiles), 3)]
w = max(r.shape[1] for r in rows); rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows]
cv2.imwrite(out, cv2.resize(np.concatenate(rows, 0), None, fx=0.55, fy=0.55))
