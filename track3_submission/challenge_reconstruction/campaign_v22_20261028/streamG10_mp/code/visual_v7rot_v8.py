"""G10 copy of H9 visual: ep32 v7_g9rot spoon (left, green) vs v8_g10 spoon (right, orange). Plain projection (numpy/cv2).
Mesh silhouette outline (green) + projected spoon ball model (red circles) + detected 2D balls (yellow crosses)."""
import sys, json, numpy as np, cv2, trimesh
from pathlib import Path
R = Path('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026')
root = R / 'canonical/evaluation/episode_000032'
OUT = Path(sys.argv[1]); frames = list(range(50, 161, 5))
intr = json.loads((root / 'intrinsics.json').read_text())
K = np.array([[intr['fx'], 0, intr['cx']], [0, intr['fy'], intr['cy']], [0, 0, 1.]])
W, H = intr['width'], intr['height']
model = np.array(json.loads((root / 'exports/spoon_roll_v8/selection.json').read_text())['objects']['wooden_spoon']['ball_roll']['model_body_m'])
dets = json.loads((root / 'ball_detections_wooden_spoon.json').read_text())['frames']
B = {}
for lab in ['selected20_final_v7_g9rot', 'selected20_final_v8_g10']:
    d = np.load(R / lab / 'episode_000032/object_poses.npz'); names = [str(n) for n in d['names']]
    B[lab] = (d['camera_from_object'][:, names.index('wooden_spoon')], d['frame_indices'])
mesh = trimesh.load(R / 'selected20_final_v6/episode_000032/meshes/wooden_spoon.obj', force='mesh', process=False)
V, F = np.asarray(mesh.vertices), np.asarray(mesh.faces)
cap = cv2.VideoCapture(str(root / 'rgb.mp4')); imgs = {}; i = 0
while True:
    ok, im = cap.read()
    if not ok: break
    if i in frames: imgs[i] = im
    i += 1
print('video frames', i, 'pose frames', len(B['selected20_final_v7_g9rot'][0]))
def proj(P, X):
    p = X @ P[:3, :3].T + P[:3, 3]; return p[:, :2] / p[:, 2:] @ K[:2, :2].T + K[:2, 2]
def draw(im, P, col):
    uv = proj(P, V); m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [np.round(uv[f]).astype(np.int32) for f in F], 1)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    ov = im.copy(); ov[m > 0] = (0.6 * ov[m > 0] + 0.4 * np.array(col)).astype(np.uint8)
    cv2.drawContours(ov, cs, -1, col, 2)
    for u in proj(P, model): cv2.circle(ov, tuple(np.round(u).astype(int)), 7, (0, 0, 255), 2)
    return ov
tiles = []; rows = []
for f in frames:
    im = imgs[f]; out = []
    for lab, col in [('selected20_final_v7_g9rot', (0, 255, 0)), ('selected20_final_v8_g10', (255, 128, 0))]:
        Pc, fi = B[lab]; t = int(np.where(fi == f)[0][0]) if f in fi else f
        o = draw(im, Pc[t], col)
        bs = dets[f] if f < len(dets) and dets[f] else []
        for b in bs: cv2.drawMarker(o, tuple(np.round(b['uv']).astype(int)), (0, 255, 255), cv2.MARKER_TILTED_CROSS, 14, 2)
        bp = np.nan
        if bs:
            uv = proj(Pc[t], model); bp = float(np.median([np.min(np.linalg.norm(uv - b['uv'], axis=1)) for b in bs]))
        # crop around spoon+hand
        c = proj(Pc[t], V).mean(0)
        x0 = int(np.clip(c[0] - 260, 0, W - 520)); y0 = int(np.clip(c[1] - 200, 0, H - 400))
        o = o[y0:y0 + 400, x0:x0 + 520].copy()
        cv2.putText(o, f'f{f} {"v6" if "v6" in lab else "v7rot"} ball {bp:.1f}px', (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        out.append(o); rows.append(dict(f=f, lab=lab, ball_px=None if np.isnan(bp) else round(bp, 1), nb=len(bs)))
    tiles.append(np.hstack(out))
pages = [tiles[i:i + 6] for i in range(0, len(tiles), 6)]
for k, pg in enumerate(pages):
    while len(pg) % 2: pg.append(np.zeros_like(pg[0]))
    sheet = np.vstack([np.hstack(pg[j:j + 2]) for j in range(0, len(pg), 2)])
    sheet = cv2.resize(sheet, None, fx=0.6, fy=0.6)
    cv2.imwrite(str(OUT / f'ep32_v7rot_vs_v8g10_{k}.png'), sheet)
json.dump(rows, open(OUT / 'ball_px.json', 'w'), indent=0)
for r in rows: print(r)
