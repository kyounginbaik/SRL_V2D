"""Marker-ball face cue (independent image evidence for half-turn questions): the scan's marker balls (scan points with no
counterpart under the half turn about the near-symmetry axis) are placed through each track's kit registration (exact
scorer seed) and projected; score = fraction landing within R px of grey, low-saturation pixels inside the dilated SAM2 mask
(the balls are grey; the sweeper is yellow/brown, the boards are light wood). Also scored for each track's half-turn twin, so
the cue's discriminating power is visible. usage: face.py EP OBJ STRIDE OUT.json [AXIS_IDX]"""
import sys, os, json, numpy as np, cv2, trimesh
sys.path.insert(0, os.path.dirname(__file__)); from common import *
import h2h
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
ep, obj, stride, out = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4]
o = load_ours(ep); n = load_new(ep, o['names'])
override = None
if ep == 25:
    B = CR / 'public_gt_check_20261020/boards/episode_000025'; override = {'wooden_piece_1': B / 'A/masks', 'wooden_piece_2': B / 'B/masks'}
ev = Evidence(ep, o['names'], override)
rv, rf = h2h.scan(obj); rc, rr = h2h.M._surface_moments(rv, rf); sm = trimesh.Trimesh(rv, rf, process=False)
pts = trimesh.sample.sample_surface(sm, 30000, seed=0)[0]; c = rv.mean(0); w, U = np.linalg.eigh(np.cov((rv - c).T))
# near-symmetry axis = the PCA axis whose half turn leaves the fewest unmatched points
cnt = []
for k in range(3):
    q = (pts - c) @ R.from_rotvec(np.pi * U[:, k]).as_matrix().T + c; d, _ = cKDTree(pts).query(q); cnt.append(((d > .004).sum(), d))
k = int(sys.argv[5]) if len(sys.argv) > 5 else int(np.argmin([x[0] for x in cnt])); d = cnt[k][1]
MK = pts[d > .004] - rc
Sw = np.eye(4); Sw[:3, :3] = R.from_rotvec(np.pi * U[:, k]).as_matrix(); Sw[:3, 3] = (c - rc) - Sw[:3, :3] @ (c - rc)   # half turn about the axis through the scan centre (scan-centred coords)
REG = {tn: h2h.registration((o['meshes'] if tn == 'ours' else n['meshes'])[obj], obj, ep, o['names'].index(obj)) for tn in ('ours', 'new')}
TR = {'ours': o['A_T_obj'][:, o['names'].index(obj)], 'new': n['A_T_obj'][obj]}
cap = cv2.VideoCapture(str(HAW / f'episode_{ep:06d}' / o['names'][0] / 'anycalib/undistorted.mp4')); rows = []
for t in range(0, min(len(TR['ours']), len(TR['new'])), stride):
    cap.set(cv2.CAP_PROP_POS_FRAMES, t); ok, img = cap.read(); m = ev.mask(obj, t)
    if not ok or m is None: continue
    if img.shape[1] != ev.size[0]: img = cv2.resize(img, ev.size)
    md = cv2.dilate(m.astype(np.uint8), np.ones((25, 25), np.uint8)) > 0
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV); grey = (hsv[..., 1] < 45) & (hsv[..., 2] > 60) & (hsv[..., 2] < 215) & md
    grey = cv2.morphologyEx(grey.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)); row = dict(frame=t, grey_px=int(grey.sum()))
    if grey.sum() >= 30:
        dist = cv2.distanceTransform((1 - grey).astype(np.uint8), cv2.DIST_L2, 5)
        for tn in TR:
            for lab, G in (('', np.eye(4)), ('_twin', Sw)):
                X = TR[tn][t] @ REG[tn] @ G; P = MK @ X[:3, :3].T + X[:3, 3]; uv = ev.project(P); okk = np.isfinite(uv).all(1)
                ui = np.round(uv[okk]).astype(int); inb = (ui[:, 0] >= 0) & (ui[:, 0] < ev.size[0]) & (ui[:, 1] >= 0) & (ui[:, 1] < ev.size[1])
                if inb.sum() >= 20: row[tn + lab] = float((dist[ui[inb, 1], ui[inb, 0]] < 10).mean())
        # do the two tracks put the marker side on the same side? angle between marker-centroid directions from the body centre
        dirs = {}
        for tn in TR:
            X = TR[tn][t] @ REG[tn]; dirs[tn] = X[:3, :3] @ (MK.mean(0))
        row['marker_dir_angle_deg'] = float(np.degrees(np.arccos(np.clip(dirs['ours'] @ dirs['new'] / np.linalg.norm(dirs['ours']) / np.linalg.norm(dirs['new']), -1, 1))))
    rows.append(row)
    print(t, {kk: (round(v, 2) if isinstance(v, float) else v) for kk, v in row.items()}, flush=True)
json.dump(dict(ep=ep, obj=obj, axis=k, n_marker_pts=int(len(MK)), rows=rows), open(out, 'w'))
