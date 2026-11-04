"""Overlay both tracks (ours green, new magenta: mesh sample points projected with each track's own camera model), ball detections
(yellow), and the marker-fit balls of each track (circles) on the evidence RGB. usage: overlay.py EP OBJ OUT.jpg f1,f2,... [FIT.json]"""
import sys, json, numpy as np, cv2
sys.path.insert(0, __file__.rsplit('/', 1)[0]); import h2hlib as H, cammodel as CM
ep, obj, out = int(sys.argv[1]), sys.argv[2], sys.argv[3]; frames = [int(x) for x in sys.argv[4].split(',')]
fit = json.load(open(sys.argv[5])) if len(sys.argv) > 5 else None
e = H.Ep(ep, nsamp=4000); fd = f'{H.HAW}/episode_{ep:06d}/{obj}/frames'
mo = lambda tr: 'new' if (tr == 'new' or ep in (15, 16)) else 'ours'
tiles = []
for f in frames:
    im = cv2.imread(f'{fd}/{f:06d}.png'); m = cv2.imread(f'{e.mask_dir[obj]}/{f:06d}.png', 0)
    if m is not None:
        m = cv2.resize(m, (e.W0, e.H0), interpolation=cv2.INTER_NEAREST); cs, _ = cv2.findContours((m > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(im, cs, -1, (255, 255, 255), 2)
    cen = []
    for tr, col in (('ours', (0, 255, 0)), ('new', (255, 0, 255))):
        C = e.cam_T(tr, obj, f); X = e.samp[(tr, obj)] @ C[:3, :3].T + C[:3, 3]; uv, _ = CM.project_full(e.cm, mo(tr), X)
        for p in uv[::3]:
            if np.all(np.isfinite(p)): cv2.circle(im, (int(p[0]), int(p[1])), 1, col, -1)
        cen.append(uv.mean(0))
        if obj == 'wooden_spoon':   # bowl opening (body -z at the bowl centre): arrow, and its camera-z sign in the label
            Q = np.array([[0.10, 0, 0], [0.10, 0, -0.045]]) @ C[:3, :3].T + C[:3, 3]; uq, _ = CM.project_full(e.cm, mo(tr), Q)
            cv2.arrowedLine(im, tuple(uq[0].astype(int)), tuple(uq[1].astype(int)), col, 5, tipLength=0.35)
            nz = -(C[:3, :3] @ np.array([0, 0, 1.]))[2]; cv2.putText(im, f'{tr} open {"TOWARD" if nz < 0 else "away"} cam ({nz:+.2f})', (int(uq[0][0]) - 150, int(uq[0][1]) + (60 if tr == 'ours' else 100)), 0, 1.0, col, 3)
        if fit:
            b = np.array(fit[tr]['balls_cm']) / 100; Xb = b @ C[:3, :3].T + C[:3, 3]; ub, _ = CM.project_full(e.cm, mo(tr), Xb)
            for p in ub: cv2.circle(im, (int(p[0]), int(p[1])), 18, col, 3)
    if fit:
        for ff, p in zip(fit['frames'], fit['uv']):
            if ff == f: cv2.circle(im, (int(p[0]), int(p[1])), 8, (0, 255, 255), -1)
    c = np.nanmean(cen, 0).astype(int); x0, y0 = int(np.clip(c[0] - 350, 0, e.W0 - 700)), int(np.clip(c[1] - 350, 0, e.H0 - 700))
    cr = im[y0:y0 + 700, x0:x0 + 700].copy(); cv2.putText(cr, f'ep{ep} f{f}', (10, 40), 0, 1.2, (0, 255, 255), 3); tiles.append(cr)
while len(tiles) % 4: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(out, cv2.resize(np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]), None, fx=0.5, fy=0.5)); print('wrote', out)
