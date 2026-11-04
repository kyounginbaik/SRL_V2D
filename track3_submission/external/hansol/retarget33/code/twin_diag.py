"""E31: per-seed kit registration (bundle -> component mesh) for selected objects: fit residual and the rotation gap to the best seed
(non-symmetric: full rotation; symmetric: signed axis angle), to see whether the off-twin seeds are a genuine alternative.
usage: twin_diag.py BUNDLE EP:OBJ ..."""
import sys, json
from pathlib import Path
import numpy as np, trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, '/data1/SRL_V2D/challenge_submissions_SRL/analysis_20261020/scripts')
import public_kit_score as K
M = K.load_metric('MP-SR')
C = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028'); CA = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates'); X = Path(sys.argv[1])
s29 = json.loads((C / 'selection_v29.json').read_text())
for key in sys.argv[2:]:
    e, name = key.split(':'); n = f'episode_{int(e):06d}'; mj = json.loads((X / n / 'meshes.json').read_text())
    b = X / n / mj[name]['file']; c = CA / s29[e] / n / 'meshes' / n / f'{name}.obj'
    mb = trimesh.load(b, force='mesh', process=False); mc = trimesh.load(c, force='mesh', process=False)
    bv, bf = np.asarray(mb.vertices, float), np.asarray(mb.faces); cv, cf = np.asarray(mc.vertices, float), np.asarray(mc.faces)
    rc, _ = M._surface_moments(cv, M._mesh_faces(cf.astype(float), len(cv)))
    tree = cKDTree(mc.sample(20000, seed=7)); bp = mb.sample(5000, seed=8)
    ax, order = K.symmetry_axis(name, cv); ax = np.asarray(ax, float); sym = bool(np.linalg.norm(ax) > 0 and int(order) != 1)
    out = []
    for seed in range(10):
        ca, R = M._register_object_body((bv, M._mesh_faces(bf.astype(float), len(bv))), (cv, M._mesh_faces(cf.astype(float), len(cv))), rc, 1000 + seed)
        t = rc - R @ ca; out.append((float(tree.query(bp @ R.T + t)[0].mean() * 1000), R))
    out.sort(key=lambda z: z[0]); R0 = out[0][1]
    gaps = [float(np.degrees(np.arccos(np.clip(np.dot(R.T @ ax, R0.T @ ax), -1, 1)))) if sym else float(np.degrees(Rot.from_matrix(R @ R0.T).magnitude())) for _, R in out]
    print(key, 'sym', sym, 'order', int(order), ' '.join(f'{f:.2f}mm/{g:.1f}deg' for (f, _), g in zip(out, gaps)), flush=True)
