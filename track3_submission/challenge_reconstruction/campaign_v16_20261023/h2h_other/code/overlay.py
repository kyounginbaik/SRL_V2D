"""Overlay both tracks on the undistorted RGB: projected-mesh outline (red = OURS, blue = NEW), SAM2 object mask (green), and
for scan objects with marker balls the scan's asymmetric marker points placed through each track's kit registration
(filled red = ours, blue rings = new). usage: overlay.py EP FRAME[,FRAME..] [OBJ]"""
import sys, os, numpy as np, cv2, trimesh
sys.path.insert(0, os.path.dirname(__file__)); from common import *
from h2h import registration
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
def marker_pts(n, axis_idx=None, thr=0.004):
    m = trimesh.load(SCANS / f'{n}.ply', force='mesh', process=False); rv = np.asarray(m.vertices); pts = trimesh.sample.sample_surface(m, 20000, seed=0)[0]; c = rv.mean(0)
    w, U = np.linalg.eigh(np.cov((rv - c).T)); k = axis_idx if axis_idx is not None else 2
    q = (pts - c) @ R.from_rotvec(np.pi * U[:, k]).as_matrix().T + c; d, _ = cKDTree(pts).query(q); rc = None
    import importlib.util
    return pts[d > thr], c
if __name__ == '__main__':
    ep = int(sys.argv[1]); frames = [int(x) for x in sys.argv[2].split(',')]; only = sys.argv[3] if len(sys.argv) > 3 else None
    o = load_ours(ep); n = load_new(ep, o['names']); ev = Evidence(ep, o['names'])
    objs = [k for k in n['names'] if (only is None or k == only)]
    REG = {(tn, k): registration((o['meshes'] if tn == 'ours' else n['meshes'])[k], k, ep, o['names'].index(k)) for tn in ('ours', 'new') for k in objs}
    MK = {}
    for k in objs:
        if k in ('mini_sweeper', 'mini_dust_pan', 'wooden_piece_1', 'wooden_piece_2'):
            P, c = marker_pts(k); import importlib
            # scan-centred coordinates expected by REG (origin = scan surface centroid)
            import h2h; rv, rf = h2h.scan(k); rc, rr = h2h.M._surface_moments(rv, rf); MK[k] = P - rc
    cap = cv2.VideoCapture(str(HAW / f'episode_{ep:06d}' / o['names'][0] / 'anycalib/undistorted.mp4'))
    os.makedirs(C16 / 'h2h_other/figures', exist_ok=True)
    for t in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, t); ok, img = cap.read()
        if img.shape[1] != ev.size[0]: img = cv2.resize(img, ev.size)
        for k in objs:
            m = ev.mask(k, t)
            if m is not None:
                cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(img, cs, -1, (0, 255, 0), 2)
            for tn, col in (('ours', (0, 0, 255)), ('new', (255, 128, 0))):
                A = o['A_T_obj'][t, o['names'].index(k)] if tn == 'ours' else n['A_T_obj'][k][t]
                s = render_sil(ev, sil_mesh((o['meshes'] if tn == 'ours' else n['meshes'])[k]), A, scale=1.0)
                if s is not None:
                    cs, _ = cv2.findContours(s.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(img, cs, -1, col, 3)
                if k in MK:
                    X = A @ REG[(tn, k)]; P = MK[k] @ X[:3, :3].T + X[:3, 3]; uv = ev.project(P)
                    for p in uv[np.isfinite(uv).all(1)][::3]:
                        p = tuple(np.round(p).astype(int))
                        if tn == 'ours': cv2.circle(img, p, 3, col, -1)
                        else: cv2.circle(img, p, 6, col, 1)
        cv2.putText(img, f'ep{ep} f{t}: red OURS, blue NEW, green SAM2 mask; dots/rings = scan marker balls via kit registration', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        ys, xs = np.nonzero(np.any([ev.mask(k, t) if ev.mask(k, t) is not None else np.zeros(ev.size[::-1], bool) for k in objs], 0))
        if len(xs):
            x0, x1 = max(xs.min() - 350, 0), min(xs.max() + 350, ev.size[0]); y0, y1 = max(ys.min() - 300, 0), min(ys.max() + 300, ev.size[1]); img = img[y0:y1, x0:x1]
        cv2.imwrite(str(C16) + f'/h2h_other/figures/ep{ep}_overlay_f{t:03d}' + (f'_{only}' if only else '') + '.jpg', img)
        print('wrote', t)
