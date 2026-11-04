"""K33-cw: build a K33 target from a coworker GT-free ep33 bundle (recon_E14_best_B2A_S1_HG{A,B}_sharpa ep33fix, 2026-11-02).
Untrusted input: npz loaded with allow_pickle=False; meshes/camera asserted byte-identical to our v14_g17 ep33 (so only the
object_poses differ). Target = world_from_object of BOTH objects (cup + spoon) from the bundle; mesh files as in build_h33.py
(cup mesh from target_final_v14_g17, spoon mesh = bundle mesh == v14). usage: cwtgt33.py LABEL BUNDLE_EP_DIR"""
import sys, json, shutil, hashlib, numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as Rot
K = Path('/data1/hansol/retarget33')
V14 = Path('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/selected20_final_v14_g17/episode_000033')
V14T = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/target_final_v14_g17/episode_000033')
lab, src = sys.argv[1], Path(sys.argv[2])
h = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
for f in ('meshes/blue_cup.obj', 'meshes/wooden_spoon.obj', 'camera.npz', 'intrinsics.json'): assert h(src / f) == h(V14 / f), f
z = np.load(src / 'object_poses.npz', allow_pickle=False); d = {k: z[k] for k in z.files}
v = np.load(V14 / 'object_poses.npz', allow_pickle=False)
assert list(d['names']) == list(v['names']) and d['world_from_object'].shape == v['world_from_object'].shape
for k in v.files:
    if k not in ('world_from_object', 'camera_from_object'): assert np.array_equal(d[k], v[k]), k
tg = K / 'targets' / lab / 'episode_000033'; tg.mkdir(parents=True)
names = list(d['names'])
for i, n in enumerate(names):
    np.savez_compressed(tg / f'{n}_trajectory.npz', T_world_obj=d['world_from_object'][:, i])
shutil.copy2(V14T / 'blue_cup_mesh_metric.obj', tg / 'blue_cup_mesh_metric.obj')
shutil.copy2(src / 'meshes' / 'wooden_spoon.obj', tg / 'wooden_spoon_mesh_metric.obj')
# sanity: our v14 target trajectory equals v14 bundle world_from_object (so the conversion is the same as build_h33)
for i, n in enumerate(names):
    t14 = np.load(V14T / f'{n}_trajectory.npz', allow_pickle=False)['T_world_obj']
    print(n, 'v14 target == v14 bundle:', np.allclose(t14, v['world_from_object'][:, i], atol=1e-9))
W, W14 = d['world_from_object'], v['world_from_object']
rep = {}
for i, n in enumerate(names):
    dp = np.linalg.norm(W[:, i, :3, 3] - W14[:, i, :3, 3], axis=1) * 100
    dr = np.degrees(Rot.from_matrix(np.einsum('tji,tjk->tik', W14[:, i, :3, :3], W[:, i, :3, :3])).magnitude())
    segs = [(0, 50), (50, 100), (100, 165), (165, 200), (200, 260), (260, 348)]
    rep[n] = {f'f{a}-{b}': [round(float(dp[a:b].mean()), 2), round(float(dr[a:b].mean()), 1), round(float(dr[a:b].max()), 1)] for a, b in segs}
    rep[n]['f0'] = [round(float(dp[0]), 4), round(float(dr[0]), 4)]
    print(n, 'vs v14 [mean cm, mean deg, max deg] per segment:', rep[n])
(tg.parent / 'CW_TARGET.json').write_text(json.dumps(dict(label=lab, source=str(src), object_poses_sha256=h(src / 'object_poses.npz'),
    ground_truth_used=False, vs_v14=rep), indent=1))
