"""Measure the round wood table in each recording: brown outline in frame 0, back-projected onto the table plane, fitted with a circle."""
import glob, json, os
import cv2, numpy as np, trimesh

Z = '/scratch2/exaFLOPs/latest_recon/v2/best_reconstruction_20261103/'
V = '/scratch2/exaFLOPs/srl_yam_run/source_videos/ego_cam_a/'
OUT = '/scratch2/exaFLOPs/srl_yam_run/source_meshes/'


def circle(p):
    A = np.c_[2 * p, np.ones(len(p))]
    s = np.linalg.lstsq(A, (p ** 2).sum(1), rcond=None)[0]
    return s[:2], float(np.sqrt(max(s[2] + s[:2] @ s[:2], 0)))


rows, tiles, table_fits = [], [], {}
for e in sorted(os.listdir(Z + 'bundles')):
    b = np.load(Z + f'bundles/{e}/object_poses.npz')
    I = json.load(open(Z + f'bundles/{e}/intrinsics.json'))
    W = b['world_from_rgb']
    names = [str(x) for x in b['names']]
    ok, im = cv2.VideoCapture(V + e + '.mp4').read()
    im = cv2.resize(im, (I['width'], I['height']))
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 0] >= 4) & (hsv[..., 0] <= 20) & (hsv[..., 1] > 110) & (hsv[..., 2] > 70) & (hsv[..., 2] < 230)).astype(np.uint8) * 255
    m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)), cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    _, lab, st, _ = cv2.connectedComponentsWithStats(m)
    k = 1 + int(np.argmax(st[1:, 4]))
    cs, _ = cv2.findContours((lab == k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=len)[:, 0]
    c = c[(c[:, 0] > 3) & (c[:, 0] < I['width'] - 4) & (c[:, 1] > 3) & (c[:, 1] < I['height'] - 4)]
    # Up comes from the hand run's gravity-aligned world, tied to this world through the shared camera.
    C = np.load(sorted(glob.glob(Z + f'hands/{e}/hawor_*_result.npz'))[0])['camera_to_world_transform']
    u = np.mean([W[i, :3, :3] @ C[int(i * (len(C) - 1) / max(1, len(W) - 1)), :3, :3].T @ np.array([0, 0, 1.]) for i in range(len(W))], 0)
    u /= np.linalg.norm(u)
    lows = []
    for nm in names:
        T = np.load(Z + f'targets/{e}/{nm}_trajectory.npz')['T_world_obj'][0]
        v = np.asarray(trimesh.load(Z + f'targets/{e}/{nm}_mesh_metric.obj', force='mesh').vertices)
        lows.append((((T[:3, :3] @ v.T).T + T[:3, 3]) @ u).min())
    table = min(lows)
    r = np.stack([(c[:, 0] - I['cx']) / I['fx'], (c[:, 1] - I['cy']) / I['fy'], np.ones(len(c))], 1)
    O, D = W[0, :3, 3], (W[0, :3, :3] @ r.T).T
    P = O + D * ((table - O @ u) / (D @ u))[:, None]
    a = np.cross(u, [1, 0, 0]); a /= np.linalg.norm(a)
    bb = np.cross(u, a)
    xy = np.stack([P @ a, P @ bb], 1)
    # One table in every recording, so its size is fixed and only the centre is fitted; racks and hands hide too much of the edge to fit both.
    R = .225
    rng_ = np.random.default_rng(0)
    gx, gy = np.meshgrid(np.arange(xy[:, 0].min() - .5, xy[:, 0].max() + .5, .005), np.arange(xy[:, 1].min() - .5, xy[:, 1].max() + .5, .005))
    cand = np.stack([gx.ravel(), gy.ravel()], 1)
    score = np.zeros(len(cand))
    for i_ in range(0, len(cand), 4000):
        score[i_:i_ + 4000] = (np.abs(np.linalg.norm(xy[None] - cand[i_:i_ + 4000, None], axis=2) - R) < .01).sum(1)
    # Objects hide much of the edge, so among the good edge fits take the one whose disc actually covers the table's brown pixels.
    ys, xs = np.nonzero(lab == k)
    pick = rng_.choice(len(xs), min(4000, len(xs)), replace=False)
    r2 = np.stack([(xs[pick] - I['cx']) / I['fx'], (ys[pick] - I['cy']) / I['fy'], np.ones(len(pick))], 1)
    D2 = (W[0, :3, :3] @ r2.T).T
    P2 = O + D2 * ((table - O @ u) / (D2 @ u))[:, None]
    br = np.stack([P2 @ a, P2 @ bb], 1)
    top_ = np.argsort(-score)[:1500]
    inside = (np.linalg.norm(br[None] - cand[top_, None], axis=2) < R).sum(1)
    c0 = cand[top_[int(np.argmax(inside + 3 * score[top_]))]]
    best = np.abs(np.linalg.norm(xy - c0, axis=1) - R) < .01
    # Refine the centre with the edge points that agree.
    for _ in range(5):
        d = xy[best] - c0
        c0 = c0 + (d - R * d / np.linalg.norm(d, axis=1, keepdims=True)).mean(0)
    center = c0[0] * a + c0[1] * bb + table * u
    table_fits[e] = dict(center_world=center.tolist(), radius=R, up_world=u.tolist(), inliers=float(best.mean()))
    rows.append((e[-3:], round(R * 100, 1), round(float(best.mean()), 2)))
    ang = np.linspace(0, 2 * np.pi, 200)
    pts = (c0[0] + R * np.cos(ang))[:, None] * a + (c0[1] + R * np.sin(ang))[:, None] * bb + table * u
    q = (pts - O) @ W[0, :3, :3]
    uv = np.stack([I['fx'] * q[:, 0] / q[:, 2] + I['cx'], I['fy'] * q[:, 1] / q[:, 2] + I['cy']], 1).astype(np.int32)
    cv2.polylines(im, [uv], True, (0, 255, 0), 3)
    cv2.putText(im, f'{e[-3:]} R={R * 100:.1f}', (10, 45), 0, 1.3, (255, 255, 255), 3)
    tiles.append(cv2.resize(im, (338, 253)))
print(rows)
print('median radius cm', np.median([x[1] for x in rows]), 'range', min(x[1] for x in rows), max(x[1] for x in rows))
json.dump(table_fits, open(OUT + 'table_fits.json', 'w'), indent=1)
cv2.imwrite(OUT + 'table_fit.png', np.vstack([np.hstack(tiles[i:i + 5]) for i in (0, 5, 10, 15)]))
print('TABLE_DONE')
