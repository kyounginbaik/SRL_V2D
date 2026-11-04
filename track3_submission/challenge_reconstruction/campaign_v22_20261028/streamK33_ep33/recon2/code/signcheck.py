"""K33-recon2 end-to-end sign check in the SUBMITTED frame: decode a package's ep33 spoon row f0 pose (+ its mesh payload), compute the
body-frame rotation vs a reference package's ep33 spoon f0 (v29), decomposed about the payload's PCA long axis (build_h33 sign rule:
largest |component| positive), and the same through the kit registration (official seed) onto the v14 bundle mesh frame.
usage: signcheck.py REF_PARQUET PARQUET [PARQUET...]"""
import sys, io, base64, numpy as np, pandas as pd, trimesh
from scipy.spatial.transform import Rotation as Rot
V14 = '/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/selected20_final_v14_g17/episode_000033/meshes/wooden_spoon.obj'
def spoon(p):
    s = pd.read_parquet(p); s = s[s.row_id.str.startswith('track_3/e000033/w000/f000000/')]
    r = s[s.row_id.str.endswith('/o1')].iloc[0]
    q = r[['quat_x', 'quat_y', 'quat_z', 'quat_w']].to_numpy(float); z = np.load(io.BytesIO(base64.b64decode(r.object_mesh_b64)), allow_pickle=False)
    return Rot.from_quat(q).as_matrix(), np.asarray(z['vertices'], float)
def long_axis(v):
    c = v - v.mean(0); w, U = np.linalg.eigh(c.T @ c / len(c)); l = U[:, 2]
    return l if l[np.argmax(np.abs(l))] > 0 else -l
R0, v0 = spoon(sys.argv[1]); L = long_axis(v0)
m = np.asarray(trimesh.load(V14, force='mesh').vertices)
print('payload vs v14 bundle mesh: max |vertex diff| (same frame?) ', 'n/a' if len(m) != len(v0) else f'{np.abs(np.sort(m, 0) - np.sort(v0, 0)).max()*1000:.2f} mm (sorted-coord)',
      ' payload long', np.round(L, 4), ' v14 long', np.round(long_axis(m), 4))
for p in sys.argv[2:]:
    R, v = spoon(p); assert np.allclose(v, v0), 'mesh payload differs'
    rv = Rot.from_matrix(R0.T @ R).as_rotvec(); ang = np.degrees(np.linalg.norm(rv))
    print(f'{p.split("/")[-3]:28s} f0 spoon vs REF: total {ang:5.2f} deg, roll about +PCA long {np.degrees(rv @ L):+6.2f} deg, off-axis {np.degrees(np.linalg.norm(rv - (rv @ L) * L)):4.2f} deg')
