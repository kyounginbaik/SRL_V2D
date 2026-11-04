"""ep20: which local difference (cup f0 depth, pitcher f0 rotation, pour window) drives the kit-level 3-4 cm between v10 and fro/c21?
Hybrids are built in v10's world: fro tracks are mapped through the frame-0 RGB camera (cam0) of each bundle."""
import numpy as np, json
from tracks import *; import kit
ep = 20
V = target('v10', ep); Fr = bundle(FRO, ep)
pv = np.load(f'{R}/reconstruction_recovery_20261026/selected20_final_v10_g12/episode_000020/object_poses.npz')
pf = np.load(f'{FRO}/episode_000020/object_poses.npz')
M_ = pv['world_from_rgb'][0] @ np.linalg.inv(pf['world_from_rgb'][0].astype(float))   # v10 world <- fro world
Fv = [(n, M_ @ T, m) for n, T, m in Fr]      # fro in v10 world, fro (our) meshes
def ang(Ra, Rb): return np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))
print('pitcher f0 rot diff v10 vs fro(mapped) deg', ang(V[1][1][0, :3, :3], Fv[1][1][0, :3, :3]).round(2))
print('cup f0 pos diff cm', ((V[0][1][0, :3, 3] - Fv[0][1][0, :3, 3]) * 100).round(2))
print('kit fro-mapped vs fro (sanity):', [x.mean(0).round(2) for x in kit.errors(Fv, Fr, ep)])
def hyb(cup_from=None, pit_from=None):
    """cup_from/pit_from: list of (f0,f1) windows where the fro-mapped pose replaces v10 (hard switch)."""
    out = []
    for k, (n, T, m) in enumerate(V):
        T = T.copy(); win = cup_from if k == 0 else pit_from
        for a, b in (win or []):
            # swap pose but keep v10 mesh: fro mesh == v10 mesh bytes? use fro's pose re-expressed through mesh body identity
            T[a:b] = Fv[k][1][a:b]
        out.append((n, T, Fr[k][2] if win else m))
    return out
H = {
 'v10': V,
 'fro_mapped': Fv,
 'cup_f0': hyb(cup_from=[(0, 1)]),
 'cup_0_50': hyb(cup_from=[(0, 50)]),
 'cup_all': hyb(cup_from=[(0, 202)]),
 'pit_f0': hyb(pit_from=[(0, 1)]),
 'pit_0_50': hyb(pit_from=[(0, 50)]),
 'pit_50_110': hyb(pit_from=[(50, 110)]),
 'pit_all': hyb(pit_from=[(0, 202)]),
 'cupf0+pitf0': hyb(cup_from=[(0, 1)], pit_from=[(0, 1)]),
 'cup0_50+pit_f0': hyb(cup_from=[(0, 50)], pit_from=[(0, 1)]),
 'cup0_50+pit50_110': hyb(cup_from=[(0, 50)], pit_from=[(50, 110)]),
}
C21 = target('c21', ep); out = {}
for k, h in H.items():
    r = {}
    for nm, ref in (('fro', Fr), ('v10', V), ('c21', C21)):
        p, o = kit.errors(h, ref, ep); r[nm] = [round(float(p[:, 0].mean()), 2), round(float(o[:, 0].mean()), 1), round(float(p[:, 1].mean()), 2), round(float(o[:, 1].mean()), 1)]
    out[k] = r; print(f'{k:20s} vs fro cup {r["fro"][0]:.2f}/{r["fro"][1]:.1f} pit {r["fro"][2]:.2f}/{r["fro"][3]:.1f} | vs v10 cup {r["v10"][0]:.2f}/{r["v10"][1]:.1f} pit {r["v10"][2]:.2f}/{r["v10"][3]:.1f} | vs c21 cup {r["c21"][0]:.2f} pit {r["c21"][2]:.2f}/{r["c21"][3]:.1f}', flush=True)
json.dump(out, open('../out/a2_hyb20.json', 'w'), indent=1)
