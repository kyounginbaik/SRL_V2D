"""Kit-faithful per-frame errors (registration with the kit seed, anchor frame-0 alignment, symmetry-aware orientation)
between a SUBMISSION source and a TARGET (compose-layout dir: <obj>_trajectory.npz + <obj>_mesh_metric.obj).
SUBMISSION source = a package dir (parquet + results) or 'target:<dir>' (a recon used as if it were the submission).
usage: perframe.py SRC TARGET EP OUT_NPZ [SEED_OVERRIDE]"""
import sys, os, glob, json
import numpy as np, pandas as pd, trimesh
sys.path.insert(0, '/data1/SRL_V2D/challenge_submissions_SRL/analysis_20261020/scripts')
import public_kit_score as K
M = K.load_metric('MP-SR')
SRC, TGT, EP, OUT = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
SEED_OFF = int(sys.argv[5]) if len(sys.argv) > 5 else 0
from scipy.spatial.transform import Rotation as Rot
def load(p, cap=40000):
    m = trimesh.load(p, force='mesh'); m.merge_vertices()
    if len(m.faces) > cap:
        try: m = m.simplify_quadric_decimation(face_count=cap)
        except Exception: pass
    m.remove_unreferenced_vertices(); return np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64)
d = f'{TGT}/episode_{EP:06d}'
if SRC.startswith('target:'):
    sd = f'{SRC[7:]}/episode_{EP:06d}'
    names = sorted([os.path.basename(p)[:-15] for p in glob.glob(f'{sd}/*_trajectory.npz')])
    # roster order: use results order of the v24 package for consistency
    r = pd.read_parquet(f'/data1/SRL_V2D/challenge_submissions_SRL/track3_v24_20261028/all_20_episodes/results/episode_{EP:06d}.parquet')
    names = [n for _, n in r.drop_duplicates('object_id').sort_values('object_id')[['object_id', 'object_body_name']].values.tolist()]
    sub_T = [np.load(f'{sd}/{n}_trajectory.npz')['T_world_obj'] for n in names]
    cands = [load(f'{sd}/{n}_mesh_metric.obj', 1 << 30) for n in names]
    nf = min(len(t) for t in sub_T)
else:
    r = pd.read_parquet(f'{SRC}/results/episode_{EP:06d}.parquet')
    names = [n for _, n in r.drop_duplicates('object_id').sort_values('object_id')[['object_id', 'object_body_name']].values.tolist()]
    sub = pd.read_parquet(f'{SRC}/track3_submission.parquet'); sub = sub[sub.row_id.str.contains(f'/e{EP:06d}/')]
    sub_T, cands = [], []
    for slot, n in enumerate(names):
        s = sub[sub.row_id.str.endswith(f'/o{slot}')].sort_values('row_id')
        T = np.tile(np.eye(4), (len(s), 1, 1)); T[:, :3, :3] = Rot.from_quat(s[['quat_x', 'quat_y', 'quat_z', 'quat_w']].values).as_matrix()
        T[:, :3, 3] = s[['pos_x', 'pos_y', 'pos_z']].values; sub_T.append(T)
        enc = s.object_mesh_b64.fillna('').astype(str); cands.append(M._decode_object_mesh(enc[enc.str.strip() != ''].iloc[0]))
    nf = min(len(t) for t in sub_T)
ref_T = [np.load(f'{d}/{n}_trajectory.npz')['T_world_obj'] for n in names]
nf = min(nf, *[len(t) for t in ref_T])
def q(T):
    qq = Rot.from_matrix(T[:, :3, :3]).as_quat(); return np.concatenate([T[:, :3, 3], qq], 1)
ach = np.stack([q(t[:nf]) for t in sub_T], 1)[:, None]; ref = np.stack([q(t[:nf]) for t in ref_T], 1)[:, None]
axes, orders, regs = [], [], []
for slot, n in enumerate(names):
    rv, rf = load(f'{d}/{n}_mesh_metric.obj'); anchor, _ = M._surface_moments(rv, rf)
    ax, o = K.symmetry_axis(n, rv)
    ca, R = M._register_object_body(cands[slot], (rv, rf), anchor, 1_000_003 * EP + 101 * slot + SEED_OFF)
    ach[:, :, slot] = M._convert_object_body(ach[:, :, slot], ca, R)
    ref[:, :, slot, :3] += np.einsum('...ij,j->...i', M._rotation_matrix(ref[:, :, slot, 3:]), anchor)
    axes.append(ax); orders.append(o); regs.append(R)
al = M._align_registered_scene(ach, ref, axes)
pos = np.linalg.norm(al[..., :3] - ref[..., :3], axis=-1)[:, 0] * 100
ori = np.degrees(M._symmetry_orientation_errors(al, ref, axes, orders))[:, 0]
np.savez(OUT, names=np.array(names), pos=pos, ori=ori, regs=np.array(regs), ach=al[:, 0], ref=ref[:, 0])
summ = {n: dict(pos=round(float(pos[:, i].mean()), 2), ori=round(float(ori[:, i].mean()), 1), ori_gt90=round(float((ori[:, i] > 90).mean()), 3),
                pos_gt10=round(float((pos[:, i] > 10).mean()), 3)) for i, n in enumerate(names)}
print(EP, json.dumps(summ))
