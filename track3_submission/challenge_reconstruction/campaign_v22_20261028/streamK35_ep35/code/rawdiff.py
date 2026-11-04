"""Raw (unaligned) rollout-vs-target pose difference per frame for the spoon, decomposed in the spoon body PCA frame.
usage: rawdiff.py PARQUET TARGET_DIR EP frames..."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK35_ep35/code')
from k35lib import *
P, TG, ep = sys.argv[1], sys.argv[2], int(sys.argv[3]); fr = [int(x) for x in sys.argv[4:]]
sub = pd.read_parquet(P); sub = sub[sub.row_id.str.startswith(f'track_3/e{ep:06d}/')].copy()
sub['f'] = sub.row_id.str.extract(r'/f(\d+)/')[0].astype(int); sub['o'] = sub.row_id.str.extract(r'/o(\d+)$')[0].astype(int)
r = pd.read_parquet(f'/data1/SRL_V2D/challenge_submissions_SRL/track3_v29_20261030/all_20_episodes/results/episode_{ep:06d}.parquet', columns=['object_id','object_body_name']).drop_duplicates().sort_values('object_id')
names = list(r.object_body_name); k = names.index('wooden_spoon'); d = f'{TG}/episode_{ep:06d}'
ref = np.load(f'{d}/wooden_spoon_trajectory.npz')['T_world_obj']; inf = mesh_info(f'{d}/wooden_spoon_mesh_metric.obj', 'wooden_spoon')
s = sub[sub.o == k].sort_values('f'); pay = s.object_mesh_b64.fillna('').astype(str); pay = pay[pay.str.strip() != ''].iloc[0]
cand = M._decode_object_mesh(pay); ca, rot = M._register_object_body(cand, (inf['v'], inf['f']), inf['anchor'], 1_000_003 * ep + 101 * k)
print('registration rot of package mesh onto target mesh (deg):', np.degrees(Rot.from_matrix(rot).magnitude()).round(2))
q = s[['quat_x','quat_y','quat_z','quat_w']].to_numpy(); Rb = Rot.from_quat(q).as_matrix() @ rot.T
U = inf['pca']
for t in fr:
    D = ref[t][:3, :3].T @ Rb[t]   # body-frame difference
    rv = U.T @ Rot.from_matrix(D).as_rotvec()
    print(f'f{t:3d} body diff (deg) about [long, mid, short] PCA axes: {np.degrees(rv).round(1)}  |total {np.degrees(np.linalg.norm(rv)):.1f}')
