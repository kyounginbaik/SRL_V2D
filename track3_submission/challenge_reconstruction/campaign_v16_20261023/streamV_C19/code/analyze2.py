import json, numpy as np, sys, mujoco
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamV_C19/code')
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/h2h_other/code')
from common19 import *
import common as HC
packed = packed_meshes(); reg = registration(packed); comp = compose_tracks()
man = json.loads((EPDIR / 'input_manifest.json').read_text()); S = np.array(man['setup_from_source'])
d = np.load(SRC / 'object_poses.npz'); W = d['world_from_object'].astype(float); Wc = d['world_from_rgb'].astype(float)
ours = {n: np.einsum('ij,tjk->tik', S, W[:, i]) for i, n in enumerate(NAMES)}
cb = {n: comp[n] @ reg[n]['M'] for n in NAMES}   # compose poses of OUR body, compose world
scene = C16 / 'common_v2_candidates/C_ep19_ilc/episode_000019/backend/evaluation/execution/scene_physics.xml'
m = mujoco.MjModel.from_xml_path(str(scene)); dd = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, dd, 0)
home = {}
for n in NAMES:
    a = m.joint(n + '_free').qposadr[0]; home[n] = to_T(dd.qpos[a:a+3][None], dd.qpos[a+3:a+7][None])[0]
def rel(T1, T2):
    D = np.linalg.inv(T1) @ T2; return round(float(np.linalg.norm(D[:3, 3]) * 100), 2), round(float(np.degrees(R.from_matrix(D[:3, :3]).magnitude())), 1)
def dlt(A, B):  # world-frame difference of two poses
    return round(float(np.linalg.norm(A[:3, 3] - B[:3, 3]) * 100), 2), round(float(np.degrees(R.from_matrix(A[:3, :3].T @ B[:3, :3]).magnitude())), 1)
for n in NAMES: print('home vs our ref f0', n, dlt(home[n], ours[n][0]), 'home z', home[n][2, 3].round(4), 'ref z', ours[n][0, 2, 3].round(4))
print('relative sweeper->dustpan f0: ours', rel(ours['mini_sweeper'][0], ours['mini_dust_pan'][0]), 'home', rel(home['mini_sweeper'], home['mini_dust_pan']))
D1 = np.linalg.inv(ours['mini_sweeper'][0]) @ ours['mini_dust_pan'][0]; D2 = np.linalg.inv(cb['mini_sweeper'][0]) @ cb['mini_dust_pan'][0]
print('rel diff ours vs compose', rel(D1, D2))
# sweeper f0 orientation difference ours vs compose in camera chain
Xc = S @ Wc[0] @ HC.A_T_rect(19)
for n in NAMES: print('camera-chain f0', n, dlt(Xc @ cb[n][0], ours[n][0]), 'mean over clip', np.mean([dlt(Xc @ np.linalg.inv(comp_c) if False else Xc @ cb[n][t], ours[n][t])[0] for t in range(235) for comp_c in [None]]))
# rotation axis of the sweeper f0 difference (in setup world)
Dsw = (Xc @ cb['mini_sweeper'][0])[:3, :3] @ ours['mini_sweeper'][0][:3, :3].T; rv = R.from_matrix(Dsw).as_rotvec(); print('sweeper f0 world rot diff axis', (rv / np.linalg.norm(rv)).round(3), np.degrees(np.linalg.norm(rv)).round(1))
Ddp = (Xc @ cb['mini_dust_pan'][0])[:3, :3] @ ours['mini_dust_pan'][0][:3, :3].T; rv = R.from_matrix(Ddp).as_rotvec(); print('dustpan f0 world rot diff axis', (rv / np.linalg.norm(rv)).round(3), np.degrees(np.linalg.norm(rv)).round(1))
# sweeper long axis in our body frame and its direction in world
v = np.asarray(packed['mini_sweeper'][0]); w, U = np.linalg.eigh(np.cov((v - v.mean(0)).T)); print('sweeper pca axes (body) eig', w.round(6)); print(U.round(3))
for n in NAMES:
    Rw = ours[n][0][:3, :3]; print(n, 'body axes in world (cols):'); print(Rw.round(3))
