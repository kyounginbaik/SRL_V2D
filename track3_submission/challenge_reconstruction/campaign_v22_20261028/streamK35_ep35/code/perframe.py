"""Per-frame kit-faithful errors of a package's ep rollout vs a recon target (same-mesh shortcut: registration of the package
mesh onto the target mesh via the kit's own _register_object_body, then kit alignment). usage: perframe.py PARQUET TARGET_DIR EP [step]"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK35_ep35/code')
from k35lib import *
P, TG, ep = sys.argv[1], sys.argv[2], int(sys.argv[3]); step = int(sys.argv[4]) if len(sys.argv) > 4 else 20
sub = pd.read_parquet(P); sub = sub[sub.row_id.str.startswith(f'track_3/e{ep:06d}/')].copy()
sub['f'] = sub.row_id.str.extract(r'/f(\d+)/')[0].astype(int); sub['o'] = sub.row_id.str.extract(r'/o(\d+)$')[0].astype(int)
r = pd.read_parquet(f'/data1/SRL_V2D/challenge_submissions_SRL/track3_v29_20261030/all_20_episodes/results/episode_{ep:06d}.parquet', columns=['object_id','object_body_name']).drop_duplicates().sort_values('object_id')
names = list(r.object_body_name); d = f'{TG}/episode_{ep:06d}'
ref = [np.load(f'{d}/{n}_trajectory.npz')['T_world_obj'] for n in names]; infos = [mesh_info(f'{d}/{n}_mesh_metric.obj', n) for n in names]
ach = []
for k, n in enumerate(names):
    s = sub[sub.o == k].sort_values('f'); pay = s.object_mesh_b64.fillna('').astype(str); pay = pay[pay.str.strip() != ''].iloc[0]
    cand = M._decode_object_mesh(pay); ca, rot = M._register_object_body(cand, (infos[k]['v'], infos[k]['f']), infos[k]['anchor'], 1_000_003 * ep + 101 * k)
    q = s[['quat_x','quat_y','quat_z','quat_w']].to_numpy(); p = s[['pos_x','pos_y','pos_z']].to_numpy()
    Rm = Rot.from_quat(q).as_matrix(); Rb = Rm @ rot.T                     # kit _convert_object_body
    pa = p + np.einsum('tij,j->ti', Rm, ca)                                # anchor point position
    T = np.tile(np.eye(4), (len(s), 1, 1)); T[:, :3, :3] = Rb; T[:, :3, 3] = pa - np.einsum('tij,j->ti', Rb, infos[k]['anchor'])
    ach.append(T[:len(ref[k])])
n = min(len(a) for a in ach + ref); pos, ori = aligned_errors([a[:n] for a in ach], [x[:n] for x in ref], infos)
print('names', names, 'mean pos', pos.mean(0).round(2), 'mean ori', ori.mean(0).round(1))
for t in range(0, n, step): print(f'f{t:3d}', ' '.join(f'{nm[:6]} {pos[t,k]:5.2f}cm {ori[t,k]:5.1f}d' for k, nm in enumerate(names)))
