"""G18 task 1 (GT-free): spoon mocap-ball CENTRES in our spoon body frame, pooled over the 10 eval spoon episodes.
Ball detections (RGB clock) + stereo camera_point (= front surface of the ball) -> centre = point + r * ray (as marker_depth.py),
r = radius_px * z / f per detection. Body frame via the RGB-clock v10_g12 spoon poses (G16 clock note). Robust mode near the
spoon_roll_v8 model seeds. Also reports the old model (front-surface median) and the signed distance of each centre to our mesh."""
import json, sys, numpy as np, trimesh
from pathlib import Path
RR = Path('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026')
EPS = [3, 5, 6, 15, 16, 32, 33, 35, 36, 37]
mesh = trimesh.load(RR / 'selected20_final_v12b_g15/episode_000032/meshes/wooden_spoon.obj', force='mesh', process=False)
model = np.array(json.loads((RR / 'canonical/evaluation/episode_000032/exports/spoon_roll_v8/selection.json').read_text())
                 ['objects']['wooden_spoon']['ball_roll']['model_body_m'])
def mode(P, seed, rad=0.012):
    c = np.array(seed, float)
    for _ in range(40):
        near = np.linalg.norm(P - c, axis=1) < rad
        if near.sum() < 3: return None, 0
        c = np.median(P[near], 0)
    return c, int(near.sum())
front, cent, rads, per_ep = [], [], [], {}
for e in EPS:
    root = RR / f'canonical/evaluation/episode_{e:06d}'
    K = json.loads((root / 'intrinsics.json').read_text()); f = np.sqrt(K['fx'] * K['fy'])
    d = np.load(RR / f'selected20_final_v10_g12/episode_{e:06d}/object_poses.npz'); names = [str(n) for n in d['names']]
    cam = d['camera_from_object'][:, names.index('wooden_spoon')].astype(float)
    det = json.loads((root / 'ball_detections_wooden_spoon.json').read_text())['frames']
    F, C, Rr = [], [], []
    for t, balls in enumerate(det[:len(cam)]):
        inv = np.linalg.inv(cam[t])
        for b in balls:
            if b.get('camera_point') is None or b.get('depth_mad_m', 1) > 0.003: continue
            p = np.array(b['camera_point']); r = b['radius_px'] * p[2] / f
            if not 0.003 < r < 0.010: continue
            q = p + p / np.linalg.norm(p) * r
            F.append(inv[:3, :3] @ p + inv[:3, 3]); C.append(inv[:3, :3] @ q + inv[:3, 3]); Rr.append(r)
    F, C = np.array(F), np.array(C)
    per_ep[e] = {k: (mode(C, s)[0].round(4).tolist() if mode(C, s)[0] is not None else None) for k, s in zip(('near', 'far', 'tip'), model)}
    front.append(F); cent.append(C); rads += Rr
front, cent = np.concatenate(front), np.concatenate(cent)
out = dict(radius_mm_median=float(np.median(rads) * 1000), radius_mm_iqr=(np.percentile(rads, [25, 75]) * 1000).round(2).tolist(), n=len(rads),
           old_model=model.tolist(), per_episode_centres=per_ep)
outer = max(mesh.split(only_watertight=False), key=lambda m: m.area)
SP, fi = trimesh.sample.sample_surface_even(outer, 200000, seed=0); SN = outer.face_normals[fi]
from scipy.spatial import cKDTree; ST = cKDTree(SP)
def sdist(c):  # positive OUTSIDE (along outward normal of nearest outer-surface sample)
    d, i = ST.query(c); return float(np.dot(c - SP[i], SN[i]))
for lab, P in (('front_median', front), ('centre', cent)):
    M = []
    for s in model:
        c, n = mode(P, s); M.append(c)
        sd = sdist(c)
        out.setdefault(lab, []).append(dict(pos=np.round(c, 4).tolist(), n=n, signed_dist_mm_pos_outside=round(sd * 1000, 2)))
    out[lab + '_arr'] = np.array(M).tolist()
print(json.dumps(out, indent=1))
json.dump(out, open(RR / 'streamG18_spoonmesh/out/ballcenters.json', 'w'), indent=1)
