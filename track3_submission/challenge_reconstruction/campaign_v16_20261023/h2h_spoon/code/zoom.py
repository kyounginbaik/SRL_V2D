"""Zoomed raw crops (no overlay) around the spoon bowl for visual roll checks, plus the same crop with both tracks' bowl-opening
arrows and predicted balls. usage: zoom.py EP OUT.jpg f1,f2,... [FIT.json]"""
import sys, json, numpy as np, cv2
sys.path.insert(0, __file__.rsplit('/', 1)[0]); import h2hlib as H, cammodel as CM
ep, out = int(sys.argv[1]), sys.argv[2]; frames = [int(x) for x in sys.argv[3].split(',')]; fit = json.load(open(sys.argv[4])) if len(sys.argv) > 4 else None
e = H.Ep(ep, nsamp=10); fd = f'{H.HAW}/episode_{ep:06d}/wooden_spoon/frames'; rows = []
for f in frames:
    im = cv2.imread(f'{fd}/{f:06d}.png'); ov = im.copy(); cs = []
    for tr, col in (('ours', (0, 255, 0)), ('new', (255, 0, 255))):
        C = e.cam_T(tr, 'wooden_spoon', f); m = e.model_of(tr)
        Q = np.array([[0.10, 0, 0], [0.10, 0, -0.045], [-0.02, 0, 0]]) @ C[:3, :3].T + C[:3, 3]; uq, _ = CM.project_full(e.cm, m, Q); cs.append(uq[[0, 2]].mean(0))
        cv2.arrowedLine(ov, tuple(uq[0].astype(int)), tuple(uq[1].astype(int)), col, 4, tipLength=0.3)
        if fit:
            b = np.array(fit[tr]['balls_cm']) / 100; ub, _ = CM.project_full(e.cm, m, b @ C[:3, :3].T + C[:3, 3])
            for p in ub: cv2.circle(ov, (int(p[0]), int(p[1])), 16, col, 3)
    c = np.mean(cs, 0).astype(int); x0, y0 = int(np.clip(c[0] - 260, 0, e.W0 - 520)), int(np.clip(c[1] - 200, 0, e.H0 - 400))
    a = im[y0:y0 + 400, x0:x0 + 520].copy(); b = ov[y0:y0 + 400, x0:x0 + 520].copy(); cv2.putText(a, f'ep{ep} f{f}', (8, 30), 0, 1, (0, 255, 255), 2)
    rows.append(np.hstack([a, b]))
cv2.imwrite(out, np.vstack(rows)); print('wrote', out)
