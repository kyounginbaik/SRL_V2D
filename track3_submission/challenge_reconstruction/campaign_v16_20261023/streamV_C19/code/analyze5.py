import json, numpy as np, sys, trimesh
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamV_C19/code')
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/h2h_other/code')
from common19 import *
import common as HC
packed = packed_meshes(); reg = registration(packed); comp = compose_tracks()
man = json.loads((EPDIR / 'input_manifest.json').read_text()); S = np.array(man['setup_from_source'])
d = np.load(SRC / 'object_poses.npz'); W = d['world_from_object'].astype(float); Wc = d['world_from_rgb'].astype(float)
ours = {n: np.einsum('ij,tjk->tik', S, W[:, i]) for i, n in enumerate(NAMES)}; cb = {n: comp[n] @ reg[n]['M'] for n in NAMES}
Vs = {n: trimesh.sample.sample_surface(trimesh.Trimesh(np.asarray(packed[n][0]), np.asarray(packed[n][1]), process=False), 300, seed=1)[0] for n in NAMES}
Vall = {n: np.asarray(packed[n][0]) for n in NAMES}
def kabsch(A, B):
    ca, cb_ = A.mean(0), B.mean(0); U, s, Vt = np.linalg.svd((A - ca).T @ (B - cb_)); D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]); Rm = Vt.T @ D @ U.T
    X = np.eye(4); X[:3, :3] = Rm; X[:3, 3] = cb_ - Rm @ ca; return X
def fitX(frames, objs=NAMES):
    A = np.concatenate([Vs[n] @ cb[n][t][:3, :3].T + cb[n][t][:3, 3] for n in objs for t in frames]); B = np.concatenate([Vs[n] @ ours[n][t][:3, :3].T + ours[n][t][:3, 3] for n in objs for t in frames]); return kabsch(A, B)
opts = dict(all=fitX(range(235)), rest=fitX(list(range(0, 15)) + list(range(205, 235))), start=fitX(range(0, 15)), end=fitX(range(205, 235)), dp_f0=ours['mini_dust_pan'][0] @ np.linalg.inv(cb['mini_dust_pan'][0]), sw_f0=ours['mini_sweeper'][0] @ np.linalg.inv(cb['mini_sweeper'][0]), cam=S @ Wc[0] @ HC.A_T_rect(19))
for k, X in opts.items():
    out = {}
    for n in NAMES:
        P = X @ cb[n]; sh = np.array([np.linalg.norm(P[t, :3, 3] - ours[n][t, :3, 3]) for t in range(235)]) * 100
        rt = np.array([np.degrees(R.from_matrix(P[t, :3, :3].T @ ours[n][t, :3, :3]).magnitude()) for t in range(235)])
        low = lambda t: (Vall[n] @ P[t, :3, :3].T + P[t, :3, 3])[:, 2].min() * 100
        out[n] = dict(shift_mean=round(sh.mean(), 2), shift_max=round(sh.max(), 1), rot_mean=round(rt.mean(), 1), f0=(round(sh[0], 2), round(rt[0], 1)), low_f0=round(low(0), 2), low_f234=round(low(234), 2), low_f10=round(low(10),2))
    print(k, json.dumps(out))
np.savez(V / 'scores/X_options.npz', **opts)
