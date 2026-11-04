"""K33-recon2 end-to-end sign check in the SUBMITTED frame (the f0 spoon pose of every rollout == v29's, so a start-roll hypothesis h acts
through the rest of the rollout): per frame t >= T0, world rotation D_t = R_pkg(t) R_ref(t)^T of the spoon, projected on the world
direction of the spoon's +PCA long axis at f0 (ref), median over frames. A hypothesis h (build_h33 convention) should read -h.
usage: signcheck2.py EP REF_PARQUET T0 PARQUET [PARQUET...]"""
import sys, io, base64, numpy as np, pandas as pd
from scipy.spatial.transform import Rotation as Rot
EP, T0 = int(sys.argv[1]), int(sys.argv[3])
def load(p):
    s = pd.read_parquet(p); s = s[s.row_id.str.startswith(f'track_3/e{EP:06d}/w000/')].copy()
    res = pd.read_parquet(p.replace('track3_submission.parquet', f'results/episode_{EP:06d}.parquet'), columns=['object_id', 'object_body_name']).drop_duplicates().sort_values('object_id')
    sl = list(res.object_body_name).index('wooden_spoon'); s = s[s.row_id.str.endswith(f'/o{sl}')]
    s['f'] = s.row_id.str.extract(r'/f(\d+)/')[0].astype(int); s = s.sort_values('f')
    R = Rot.from_quat(s[['quat_x', 'quat_y', 'quat_z', 'quat_w']].to_numpy(float)).as_matrix()
    z = np.load(io.BytesIO(base64.b64decode(s.object_mesh_b64.iloc[0])), allow_pickle=False); v = np.asarray(z['vertices'], float)
    c = v - v.mean(0); w, U = np.linalg.eigh(c.T @ c / len(c)); L = U[:, 2]; L = L if L[np.argmax(np.abs(L))] > 0 else -L
    return R, L, s[['pos_x', 'pos_y', 'pos_z']].to_numpy(float)
Rr, L, Pr = load(sys.argv[2]); a = Rr[0] @ L
for p in sys.argv[4:]:
    R, _, P = load(p)
    rv = np.array([Rot.from_matrix(R[t] @ Rr[t].T).as_rotvec() for t in range(T0, len(R))])
    roll = np.degrees(rv @ a); tot = np.degrees(np.linalg.norm(rv, axis=1))
    print(f'{p.split("/")[-3]:28s} f0 diff {np.degrees(np.linalg.norm(Rot.from_matrix(R[0] @ Rr[0].T).as_rotvec())):.2f} deg | f>={T0}: world rot about f0 long median {np.median(roll):+6.2f} (IQR {np.percentile(roll,25):+.1f}..{np.percentile(roll,75):+.1f}), total median {np.median(tot):.1f}')
