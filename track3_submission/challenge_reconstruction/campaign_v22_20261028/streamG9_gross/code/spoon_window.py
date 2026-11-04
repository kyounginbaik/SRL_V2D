"""GT-free arbitration of ours-vs-compose spoon roll disagreements. For every frame: our final_v6 pose vs the compose pose
mapped into our camera/body frame through the kit registration + frame-0 alignment (the same mapping the scorer uses);
evidence = 2D ball reprojection (median px, spoon_roll_v8 ball model) and visible-mask IoU (Z9 evidence()).
usage: spoon_window.py EP TARGET_LABEL TARGET_DIR OUT_JSON"""
import sys, json, numpy as np, trimesh
from pathlib import Path
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamZ9_anchor_spoon/code')
from z9_anchor import frame_inputs, load_k, ball_px, ball_model, evidence
sys.path.insert(0, '/data1/SRL_V2D/challenge_submissions_SRL/analysis_20261020/scripts')
import public_kit_score as K
from scipy.spatial.transform import Rotation as Rot
M = K.load_metric('MP-SR')
EP, TL, TD, OUT = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
OBJ = sys.argv[5] if len(sys.argv) > 5 else 'wooden_spoon'
R = Path('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026')
root = R / f'canonical/evaluation/episode_{EP:06d}'; fol = R / f'selected20_final_v6/episode_{EP:06d}'
d = np.load(fol / 'object_poses.npz'); names = [str(n) for n in d['names']]
order = names  # roster order of the bundle (kit slot = same order as results in our packages)
def L(p):
    m = trimesh.load(p, force='mesh'); return np.asarray(m.vertices, float), np.asarray(m.faces, np.int64)
Wo = d['world_from_object']; nf = len(Wo)
ref_T = [np.load(f'{TD}/episode_{EP:06d}/{n}_trajectory.npz')['T_world_obj'] for n in names]; nf = min(nf, *[len(t) for t in ref_T])
def q(T): return np.concatenate([T[:, :3, 3], Rot.from_matrix(T[:, :3, :3]).as_quat()], 1)
ach = np.stack([q(Wo[:nf, i]) for i in range(len(names))], 1)[:, None]; ref = np.stack([q(t[:nf]) for t in ref_T], 1)[:, None]
axes, regs = [], []
for slot, n in enumerate(names):
    rv, rf = L(f'{TD}/episode_{EP:06d}/{n}_mesh_metric.obj'); anchor, _ = M._surface_moments(rv, rf)
    ax, o = K.symmetry_axis(n, rv); cand = L(fol / 'meshes' / f'{n}.obj')
    ca, Rb = M._register_object_body(cand, (rv, rf), anchor, 1_000_003 * EP + 101 * slot)
    ach[:, :, slot] = M._convert_object_body(ach[:, :, slot], ca, Rb)
    ref[:, :, slot, :3] += np.einsum('...ij,j->...i', M._rotation_matrix(ref[:, :, slot, 3:]), anchor)
    axes.append(ax); regs.append((ca, Rb))
al = M._align_registered_scene(ach, ref, axes)[:, 0]; ref = ref[:, 0]
s = names.index(OBJ); ca, Rb = regs[s]
def T(p):
    X = np.tile(np.eye(4), (len(p), 1, 1)); X[:, :3, :3] = Rot.from_quat(p[:, 3:]).as_matrix(); X[:, :3, 3] = p[:, :3]; return X
A, B = T(al[:, s]), T(ref[:, s])
Mo = np.eye(4); Mo[:3, :3] = Rb; Mo[:3, 3] = -Rb @ ca          # our body -> compose (anchored) body
cam = d['camera_from_object'][:nf, s]
implied_al = cam @ np.linalg.inv(Mo) @ np.linalg.inv(A) @ B @ Mo   # via scorer alignment (biased by f0 alignment)
cm = np.load(f'{TD}/episode_{EP:06d}/camera_motion.npz'); anc = M._surface_moments(*L(f'{TD}/episode_{EP:06d}/{OBJ}_mesh_metric.obj'))[0]
Mo2 = np.eye(4); Mo2[:3, :3] = Rb; Mo2[:3, 3] = anc - Rb @ ca      # our body -> compose mesh coords
implied = cm['T_a_from_rectL'] @ np.linalg.inv(cm['cam0_T_cam'][:nf]) @ ref_T[s][:nf] @ Mo2   # compose pose in camera a (= our RGB camera)
ori = np.degrees(Rot.from_matrix(np.linalg.inv(A)[:, :3, :3] @ B[:, :3, :3]).magnitude())
mesh = trimesh.load(fol / f'meshes/{OBJ}.obj', force='mesh', process=False)
pts, fc = trimesh.sample.sample_surface(mesh, 20000, seed=260126); nrm = np.asarray(mesh.face_normals)[fc]
k, intr = load_k(root, 507, 380); kf = np.array([[intr['fx'], 0, intr['cx']], [0, intr['fy'], intr['cy']], [0, 0, 1.]])
model, dets = ball_model(root, OBJ)
if model is None: dets = []
THR = 90 if OBJ == 'wooden_spoon' else 0
rows = []
for t in range(0, nf):
    nb = len(dets[t] or []) if t < len(dets) else 0
    disagree = bool(ori[t] > THR)
    if not disagree and t % 3: continue
    fi = frame_inputs(root, OBJ, names, t, 507, 380)
    r = dict(f=t, ori=round(float(ori[t]), 1), nb=nb)
    for lab, P in [('ours', cam[t]), (TL, implied[t])]:
        r[f'{lab}_ball'] = None if nb == 0 else round(ball_px(model, dets[t], P, kf), 1)
        if fi is not None:
            m, ex, dp, hpx = fi; _, info = evidence(mesh, pts, nrm, P, k, m, ex, dp); r[f'{lab}_iou'] = round(float(info['visible_iou']), 3); r['hand'] = hpx; r[f'{lab}_z'] = None if info.get('median_z_mm') is None else round(float(info['median_z_mm']), 1)
    rows.append(r)
dis = [r for r in rows if r['ori'] > THR]
def agg(rs, lab, key):
    v = [r.get(f'{lab}_{key}') for r in rs if r.get(f'{lab}_{key}') is not None and np.isfinite(r.get(f'{lab}_{key}'))]
    return (round(float(np.median(v)), 3), len(v)) if v else (None, 0)
summ = dict(ep=EP, target=TL, frames=nf, disagree_frac=round(float((ori > 90).mean()), 3),
            disagree=dict(ours_ball=agg(dis, 'ours', 'ball'), comp_ball=agg(dis, TL, 'ball'), ours_iou=agg(dis, 'ours', 'iou'), comp_iou=agg(dis, TL, 'iou'),
                          ball_frames_ours_better=int(sum(1 for r in dis if r.get('ours_ball') is not None and np.isfinite(r['ours_ball']) and r['ours_ball'] < r[f'{TL}_ball'])),
                          ball_frames_comp_better=int(sum(1 for r in dis if r.get('ours_ball') is not None and np.isfinite(r['ours_ball']) and r['ours_ball'] > r[f'{TL}_ball'])),
                          iou_frames_ours_better=int(sum(1 for r in dis if 'ours_iou' in r and r['ours_iou'] > r[f'{TL}_iou'] + 0.02)),
                          iou_frames_comp_better=int(sum(1 for r in dis if 'ours_iou' in r and r['ours_iou'] + 0.02 < r[f'{TL}_iou']))),
            depth=dict(ours=agg(dis, 'ours', 'z'), comp=agg(dis, TL, 'z')), agree_sample=dict(ours_ball=agg([r for r in rows if r['ori'] <= THR], 'ours', 'ball'), comp_ball=agg([r for r in rows if r['ori'] <= THR], TL, 'ball')))
print(json.dumps(summ))
json.dump(dict(summary=summ, rows=rows), open(OUT, 'w'))
