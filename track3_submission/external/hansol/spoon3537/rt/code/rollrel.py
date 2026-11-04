"""K33: signed body-frame rotation of a package's ep33 spoon rollout relative to a recon target, per frame (raw world frames; the
physics world shares the recon gravity/world frame), decomposed into roll about the spoon long PCA axis + residual tilt.
Uses the kit registration (official seed) to express the package spoon in the target mesh frame. GT-free (both ours).
usage: rollrel.py PARQUET TARGET_DIR [step]"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK35_ep35/code')
from k35lib import *
P, TG = sys.argv[1], sys.argv[2]; step = int(sys.argv[3]) if len(sys.argv) > 3 else 10; ep = 33
sub = pd.read_parquet(P); sub = sub[sub.row_id.str.startswith(f'track_3/e{ep:06d}/')].copy()
sub['f'] = sub.row_id.str.extract(r'/f(\d+)/')[0].astype(int); sub['o'] = sub.row_id.str.extract(r'/o(\d+)$')[0].astype(int)
names = ['blue_cup', 'wooden_spoon']; k = 1; d = f'{TG}/episode_{ep:06d}'
ref = np.load(f'{d}/wooden_spoon_trajectory.npz')['T_world_obj']; inf = mesh_info(f'{d}/wooden_spoon_mesh_metric.obj', 'wooden_spoon')
long = inf['pca'][:, 0]
s = sub[sub.o == k].sort_values('f'); pay = s.object_mesh_b64.fillna('').astype(str); pay = pay[pay.str.strip() != ''].iloc[0]
cand = M._decode_object_mesh(pay); ca, rot = M._register_object_body(cand, (inf['v'], inf['f']), inf['anchor'], 1_000_003 * ep + 101 * k)
Rm = Rot.from_quat(s[['quat_x','quat_y','quat_z','quat_w']].to_numpy()).as_matrix(); Rb = Rm @ rot.T
n = min(len(Rb), len(ref))
# kit alignment first (physics world vs recon world differ by a rigid world transform): anchor = spoon (no symmetry axis)
cup = np.load(f'{d}/blue_cup_trajectory.npz')['T_world_obj']; cinf = mesh_info(f'{d}/blue_cup_mesh_metric.obj', 'blue_cup')
c = sub[sub.o == 0].sort_values('f'); cpay = c.object_mesh_b64.fillna('').astype(str); cpay = cpay[cpay.str.strip() != ''].iloc[0]
cca, crot = M._register_object_body(M._decode_object_mesh(cpay), (cinf['v'], cinf['f']), cinf['anchor'], 1_000_003 * ep)
def track(rows, ca_, rot_, inf_):
    Rm_ = Rot.from_quat(rows[['quat_x','quat_y','quat_z','quat_w']].to_numpy()).as_matrix(); Rb_ = Rm_ @ rot_.T
    pa = rows[['pos_x','pos_y','pos_z']].to_numpy() + np.einsum('tij,j->ti', Rm_, ca_)
    T = np.tile(np.eye(4), (len(rows), 1, 1)); T[:, :3, :3] = Rb_; T[:, :3, 3] = pa - np.einsum('tij,j->ti', Rb_, inf_['anchor']); return T
A = [track(c, cca, crot, cinf)[:n], track(s, ca, rot, inf)[:n]]; Rf = [cup[:n], ref[:n]]; infos = [cinf, inf]
a = M._align_registered_scene(tensor(A, infos), tensor(Rf, infos), [i['axis'] for i in infos])
Ra = Rot.from_quat(a[:, 0, 1, 3:]).as_matrix(); rolls, tilts = [], []
for t in range(n):
    E = Rot.from_matrix(ref[t, :3, :3].T @ Ra[t]); rv = E.as_rotvec(); rolls.append(np.degrees(rv @ long))
    tilts.append(np.degrees(np.linalg.norm(rv - (rv @ long) * long)))
rolls = np.array(rolls); tilts = np.array(tilts)
for a, b in [(0, 25), (5, 50), (25, 52), (300, 347), (320, 347)]:
    print(f'f{a}-{b}: roll median {np.median(rolls[a:b+1]):+.1f} (p10 {np.percentile(rolls[a:b+1],10):+.1f} p90 {np.percentile(rolls[a:b+1],90):+.1f}) tilt median {np.median(tilts[a:b+1]):.1f}')
print(' '.join(f'{t}:{rolls[t]:+.0f}/{tilts[t]:.0f}' for t in range(0, n, step)))
