"""Is a constant position discrepancy (sub vs target) fixed in the object's BODY frame (= mesh/anchor registration offset) or in
the WORLD (= placement)? Kit-faithful alignment; prints mean/std of the error vector in both frames per object.
usage: errframe.py EP PKG(v24|v25) TARGET_DIR"""
import sys, numpy as np
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG10_mp/code')
from g10lib import *
EP, PK, TD = int(sys.argv[1]), sys.argv[2], sys.argv[3]
names, subT, cands = load_sub(PKG[PK], EP); refT, rm = load_target(TD, EP, names)
S = Scorer(EP, names, cands, rm)
nf = min(min(len(t) for t in subT), min(len(t) for t in refT))
ach = np.stack([S.q(t[:nf]) for t in subT], 1)[:, None]; ref = np.stack([S.q(t[:nf]) for t in refT], 1)[:, None]
for slot in range(len(names)):
    ca, R = S.regs[slot]
    ach[:, :, slot] = M._convert_object_body(ach[:, :, slot], ca, R)
    ref[:, :, slot, :3] += np.einsum('...ij,j->...i', M._rotation_matrix(ref[:, :, slot, 3:]), S.anchor[slot])
al = M._align_registered_scene(ach, ref, S.axes)[:, 0]; ref = ref[:, 0]
for i, n in enumerate(names):
    e = al[:, i, :3] - ref[:, i, :3]
    Rr = M._rotation_matrix(ref[:, i, 3:]); eb = np.einsum('fji,fj->fi', Rr, e)
    print(n, 'reg R deg', round(float(np.degrees(np.arccos(np.clip((np.trace(S.regs[i][1]) - 1) / 2, -1, 1)))), 1),
          '| world err mean cm', np.round(e.mean(0) * 100, 2), 'std', np.round(e.std(0) * 100, 2), '| body err mean cm', np.round(eb.mean(0) * 100, 2), 'std', np.round(eb.std(0) * 100, 2),
          '| |e| mean', round(float(np.linalg.norm(e, axis=1).mean() * 100), 2))
