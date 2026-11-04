"""GT-free evidence, whole clip, for one object: bundle A vs bundle B (ball px 2D, visible IoU, median |depth| residual, rotation steps).
usage: evid_compare.py EP OBJ BUNDLE_A BUNDLE_B OUT_JSON"""
import sys, json, numpy as np, trimesh
from pathlib import Path
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamZ9_anchor_spoon/code')
from z9_anchor import frame_inputs, load_k, ball_px, ball_model, evidence
from scipy.spatial.transform import Rotation as Rot
EP, OBJ, A, B, OUT = int(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]), sys.argv[5]
RR = Path('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026'); root = RR / f'canonical/evaluation/episode_{EP:06d}'
res = {}
cams = {}
for lab, bd in [('A', A), ('B', B)]:
    d = np.load(bd / f'episode_{EP:06d}/object_poses.npz'); names = [str(n) for n in d['names']]; cams[lab] = d['camera_from_object'][:, names.index(OBJ)]
mesh = trimesh.load(A / f'episode_{EP:06d}/meshes/{OBJ}.obj', force='mesh', process=False)
pts, fc = trimesh.sample.sample_surface(mesh, 20000, seed=260126); nrm = np.asarray(mesh.face_normals)[fc]
k, intr = load_k(root, 507, 380); kf = np.array([[intr['fx'], 0, intr['cx']], [0, intr['fy'], intr['cy']], [0, 0, 1.]])
model, dets = ball_model(root, OBJ)
changed = [f for f in range(len(cams['A'])) if not np.allclose(cams['A'][f], cams['B'][f])]
rows = []
for f in changed:
    r = dict(f=f); fi = frame_inputs(root, OBJ, names, f, 507, 380)
    for lab in 'AB':
        P = cams[lab][f]
        if model is not None and f < len(dets) and dets[f]: r[f'{lab}_ball'] = ball_px(model, dets[f], P, kf)
        if fi is not None:
            m, ex, dp, _ = fi; info = evidence(mesh, pts, nrm, P, k, m, ex, dp)[1]; r[f'{lab}_iou'] = float(info['visible_iou']); r[f'{lab}_z'] = info['median_z_mm']
    rows.append(r)
def med(key): v = [r[key] for r in rows if r.get(key) is not None and np.isfinite(r[key])]; return (round(float(np.median(v)), 3), len(v)) if v else (None, 0)
def steps(C): s = np.degrees((Rot.from_matrix(C[1:, :3, :3]) * Rot.from_matrix(C[:-1, :3, :3]).inv()).magnitude()); return dict(gt30=int((s > 30).sum()), max=round(float(s.max()), 1))
summ = dict(ep=EP, obj=OBJ, changed_frames=len(changed), A=dict(ball=med('A_ball'), iou=med('A_iou'), z_mm=med('A_z'), steps=steps(cams['A'])),
            B=dict(ball=med('B_ball'), iou=med('B_iou'), z_mm=med('B_z'), steps=steps(cams['B'])),
            ball_B_better=int(sum(1 for r in rows if 'A_ball' in r and r['B_ball'] < r['A_ball'])), ball_A_better=int(sum(1 for r in rows if 'A_ball' in r and r['B_ball'] > r['A_ball'])),
            iou_B_better=int(sum(1 for r in rows if 'A_iou' in r and r['B_iou'] > r['A_iou'] + 0.02)), iou_A_better=int(sum(1 for r in rows if 'A_iou' in r and r['B_iou'] + 0.02 < r['A_iou'])))
print(json.dumps(summ)); json.dump(dict(summary=summ, rows=rows), open(OUT, 'w'))
