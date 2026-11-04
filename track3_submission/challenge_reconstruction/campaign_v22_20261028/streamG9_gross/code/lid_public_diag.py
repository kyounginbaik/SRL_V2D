"""SCORER-SIDE diagnosis only (public GT): is our lid orientation error a constant registration spin or tracking?"""
import sys, json, numpy as np, trimesh
sys.path.insert(0, '/data1/SRL_V2D/challenge_submissions_SRL/analysis_20261020/scripts')
import public_kit_score as K
from scipy.spatial.transform import Rotation as Rot
M = K.load_metric('MP-SR')
R = '/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/canonical/public_heldout'
def L(p):
    m = trimesh.load(p, force='mesh'); m.merge_vertices(); return np.asarray(m.vertices, float), np.asarray(m.faces, np.int64)
scan = L(K.gt_mesh_path('white_pot_lid')); rc, _ = M._surface_moments(*scan); ax, _ = K.symmetry_axis('white_pot_lid', scan[0])
print('scan lid axis', ax.round(3))
for ep in [0, 2]:
    d = f'{R}/episode_{ep:06d}'; z = np.load(f'{d}/exports/r8_base/object_poses.npz'); names = [str(n) for n in z['names']]; i = names.index('white_pot_lid')
    ours = z['world_from_object'][:, i]; gt, vis = K.gt_world_T_obj(ep, 'white_pot_lid'); n = min(len(ours), len(gt))
    mj = json.load(open(f'{d}/exports/r8_base/meshes.json')); mp = mj['white_pot_lid']
    cand = L(mp)
    ca, Rb = M._register_object_body(cand, scan, rc, 1_000_003 * ep + 101 * i)
    # per-frame implied registration: R_gt^T R_ours (maps our body -> scan body if ours were exact, up to world rot)
    Rconv = np.einsum('fji,fjk->fik', gt[:n, :3, :3], ours[:n, :3, :3])
    # remove the world alignment: kit aligns with the anchor (pot) at frame 0; use frame-relative comparison
    rel = [Rot.from_matrix(Rb @ Rconv[f].T) for f in range(n)]  # Rb vs Rconv(f)
    ang = np.array([r.magnitude() for r in rel]) * 57.3
    # spin component about the scan axis
    spin = np.array([np.degrees(np.arctan2(*(lambda v: (v @ np.cross(ax, [1, 0, 0]) if abs(ax[0]) < .9 else v @ np.cross(ax, [0, 1, 0]), 0))(r.as_rotvec())) ) for r in rel])
    rv = np.array([r.as_rotvec() for r in rel]); along = np.degrees(rv @ ax); perp = np.degrees(np.linalg.norm(rv - np.outer(rv @ ax, ax), axis=1))
    print(ep, 'n', n, 'Rb vs per-frame convention: total deg median %.1f p10 %.1f p90 %.1f | about-axis median %.1f (p10 %.1f p90 %.1f) | tilt median %.1f' % (np.median(ang), np.percentile(ang, 10), np.percentile(ang, 90), np.median(along), np.percentile(along, 10), np.percentile(along, 90), np.median(perp)))
