"""Hypothesis-only (nothing copied into any target): which part of the compose ep20 hypothesis makes packages fail?
Truth hybrids in v10's world (via cam0): (cup from X, pitcher from Y), X,Y in {v10, c21}; c21 pitcher mesh = its scan mesh, so this is
scorer-side reasoning only. Report package outcomes (first ~ old, v26 ~ new) per hybrid."""
import numpy as np
from tracks import *; import kit
ep = 20
V = target('v10', ep); Cc = target('c21', ep)
pv = np.load(f'{R}/reconstruction_recovery_20261026/selected20_final_v10_g12/episode_000020/object_poses.npz')
cm = np.load(f'{TGT["c21"]}/episode_000020/camera_motion.npz')
Mp = pv['world_from_rgb'][0] @ cm['T_a_from_rectL']      # v10 world <- compose world (rectL0)
Cm = [(n, Mp @ T, m) for n, T, m in Cc]
print('sanity c21-mapped vs c21:', [x.mean(0).round(2) for x in kit.errors(Cm, Cc, ep)])
H = {'v10cup+v10pit': V, 'c21cup+c21pit': Cm, 'v10cup+c21pit': [V[0], Cm[1]], 'c21cup+v10pit': [Cm[0], V[1]]}
# pour-window-only variant: v10 everywhere, c21 pitcher only in f45-115 (hard switch)
T = V[1][1].copy(); 
for f in range(45, 115): T[f] = Cm[1][1][f] @ np.linalg.inv(Cm[1][1][44]) @ V[1][1][44] if False else T[f]
for k, h in H.items():
    s = []
    for p in ['first', 'v21C', 'v22b', 'v26']:
        pos, ori = kit.errors(package(p, ep), h, ep); pm, om = pos.mean(0), ori.mean(0)
        s.append(f'{p}: cup {pm[0]:.2f}/{om[0]:.1f} pit {pm[1]:.2f}/{om[1]:.1f} MP{int(np.all((pm<3)&(om<30)))}')
    print(f'{k:16s} ' + ' | '.join(s), flush=True)
