"""ep20 cup: surface-centroid (kit anchor) comparison v10 vs fro in our camera, and per-frame registration of 277d45fc (fro/package cup) to 415ade56 (v10 cup)."""
import numpy as np, trimesh, json
from tracks import *; import kit
M = kit.M
B = f'{R}/reconstruction_recovery_20261026/selected20_final_v10_g12/episode_000020'
pv = np.load(f'{B}/object_poses.npz'); Cv = pv['camera_from_object']
Cf = np.stack([np.load(f'{FRO}/episode_000020/{n}_trajectory.npz')['T_rgb_obj'] for n in ['blue_cup', 'water_pitcher']], 1)
def sc(path):
    m = trimesh.load(path, force='mesh'); v, f = np.asarray(m.vertices), np.asarray(m.faces); c, r = M._surface_moments(v, f); return c, r, v, f
cv, rv, Vv, Fv = sc(f'{B}/meshes/blue_cup.obj'); cf, rf, Vf, Ff = sc(f'{FRO}/episode_000020/assets/blue_cup.obj')
print('surface centroid / radius v10 cup', cv.round(4), rv, ' fro cup', cf.round(4), rf)
ax_v = K.symmetry_axis('blue_cup', Vv)[0] if False else None
import public_kit_score as K
av = K.symmetry_axis('blue_cup', Vv)[0]; af = K.symmetry_axis('blue_cup', Vf)[0]
for nm, (cc, ca, CC) in {'v10': (cv, av, Cv[:, 0]), 'fro': (cf, af, Cf[:, 0])}.items(): pass
pv_ = Cv[:, 0, :3, :3] @ cv + Cv[:, 0, :3, 3]; pf_ = Cf[:, 0, :3, :3] @ cf + Cf[:, 0, :3, 3]
d = (pv_ - pf_); ray = pv_ / np.linalg.norm(pv_, axis=1)[:, None]; al = (d * ray).sum(1); perp = np.linalg.norm(d - al[:, None] * ray, axis=1)
axv = Cv[:, 0, :3, :3] @ av; axf = Cf[:, 0, :3, :3] @ af; tilt = np.degrees(np.arccos(np.clip(np.abs((axv * axf).sum(1)), 0, 1)))
for a, b in [(0, 1), (1, 10), (10, 30), (30, 50), (50, 80), (80, 110), (110, 150), (150, 202)]:
    print(f'cup f{a:3d}-{b:3d}: |d| {np.linalg.norm(d[a:b],axis=1).mean()*100:.2f} along(v10-fro) {al[a:b].mean()*100:+.2f} perp {perp[a:b].mean()*100:.2f} axis {tilt[a:b].mean():.1f} deg  z v10 {pv_[a:b,2].mean()*100:.1f}')
# pitcher (same mesh file): surface centroid
cp, rp, _, _ = sc(f'{B}/meshes/water_pitcher.obj')
qv = Cv[:, 1, :3, :3] @ cp + Cv[:, 1, :3, 3]; qf = Cf[:, 1, :3, :3] @ cp + Cf[:, 1, :3, 3]; d = qv - qf
ray = qv / np.linalg.norm(qv, axis=1)[:, None]; al = (d * ray).sum(1); perp = np.linalg.norm(d - al[:, None] * ray, axis=1)
Rr = np.einsum('fji,fjk->fik', Cv[:, 1, :3, :3], Cf[:, 1, :3, :3]); ang = np.degrees(np.arccos(np.clip((np.trace(Rr, axis1=1, axis2=2) - 1) / 2, -1, 1)))
for a, b in [(0, 1), (1, 10), (10, 30), (30, 50), (50, 80), (80, 110), (110, 150), (150, 202)]:
    print(f'pit f{a:3d}-{b:3d}: |d| {np.linalg.norm(d[a:b],axis=1).mean()*100:.2f} along {al[a:b].mean()*100:+.2f} perp {perp[a:b].mean()*100:.2f} rot {ang[a:b].mean():.1f}  z {qv[a:b,2].mean()*100:.1f}  vec {(d[a:b].mean(0)*100).round(2)}')
