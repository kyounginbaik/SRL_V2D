"""Detect the grey mocap balls (grey S<SMAX(40), 80<V<220, not hand, compact blob 120-1800 px with a dark mount (V<70) within 15 px; tuned from streamR_ep32 spoon_markers.py) in the AnyCalib-undistorted
RGB frames, inside the object's SAM2 mask dilated by 41 px (track-independent ROI), excluding the other objects' masks.
usage: detect_balls.py EP OBJ OUT.json"""
import sys, json, numpy as np, cv2, os
sys.path.insert(0, __file__.rsplit('/', 1)[0]); import h2hlib as H
ep, obj, out = int(sys.argv[1]), sys.argv[2], sys.argv[3]
e = H.Ep(ep, nsamp=10); fd = f'{H.HAW}/episode_{ep:06d}/wooden_spoon/frames' if 'wooden_spoon' in e.names else f'{H.HAW}/episode_{ep:06d}/{e.names[0]}/frames'
def fullmask(o, f):
    m = cv2.imread(f'{e.mask_dir[o]}/{f:06d}.png', 0)
    if m is None: return np.zeros((e.H0, e.W0), bool)
    return cv2.resize((m > 0).astype(np.uint8), (e.W0, e.H0), interpolation=cv2.INTER_NEAREST) > 0
rows = []
SMAX = int(os.environ.get('SMAX', 40)); KIND = os.environ.get('KIND', 'ball')
def handfull(f):
    h = np.zeros((e.H0, e.W0), bool)
    for d in e.hand_dirs:
        m = cv2.imread(f'{d}/{f:06d}.png', 0)
        if m is not None: h |= cv2.resize((m > 0).astype(np.uint8), (e.W0, e.H0), interpolation=cv2.INTER_NEAREST) > 0
    return h
for f in range(e.N):
    im = cv2.imread(f'{fd}/{f:06d}.png'); m = fullmask(obj, f); handm = handfull(f)
    roi = cv2.dilate(m.astype(np.uint8), np.ones((41, 41), np.uint8)) > 0
    for o in e.names:
        if o != obj: roi &= ~fullmask(o, f)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV); h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    g = ((s < SMAX) & (v > 80) & (v < 220) & roi & ~handm).astype(np.uint8) if KIND == 'ball' else ((v < 55) & roi & ~handm).astype(np.uint8)
    g = cv2.morphologyEx(g, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)); g = cv2.morphologyEx(g, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    dark = (v < 80).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(g); dets = []
    for i in range(1, n):
        x, y, w, hh, area = st[i]
        if not (120 < area < 1800 and 0.5 < w / hh < 2.0 and area / (w * hh) > 0.5): continue
        bl = (lab[max(y-16,0):y+hh+16, max(x-16,0):x+w+16] == i).astype(np.uint8); ring = cv2.dilate(bl, np.ones((31, 31), np.uint8)) - bl
        if KIND == 'ball' and (ring * dark[max(y-16,0):y+hh+16, max(x-16,0):x+w+16]).sum() < 20: continue
        if KIND == 'mount' and not (60 < area < 1500): continue
        dets.append(dict(px=[float(cen[i][0]), float(cen[i][1])], r=float(max(w, hh) / 2), area=int(area)))
    rows.append(dict(frame=f, dets=dets))
json.dump(dict(ep=ep, obj=obj, kind=KIND, frames_dir=fd, rows=rows), open(out, 'w'))
print(ep, obj, 'frames with >=1/>=2/>=3 detections', sum(len(r['dets']) >= 1 for r in rows), sum(len(r['dets']) >= 2 for r in rows), sum(len(r['dets']) >= 3 for r in rows), 'of', e.N)
