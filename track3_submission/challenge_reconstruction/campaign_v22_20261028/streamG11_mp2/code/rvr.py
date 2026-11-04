"""recon-vs-recon per-frame (kit-faithful: our mesh registered onto the other's mesh with the kit seed, anchor frame-0 alignment).
usage: rvr.py EP OURS_TARGET_DIR OTHER_TARGET_DIR [every]  -> prints per-frame aligned error vectors (cm) per object + summary.
Also prints the frame-0 relative vector between the objects (other - slot0) in each recon's own frame-0 anchor frame."""
import sys, numpy as np
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG10_mp/code')
from g10lib import *
EP, A, B = int(sys.argv[1]), sys.argv[2], sys.argv[3]; ev = int(sys.argv[4]) if len(sys.argv) > 4 else 10
import glob as _g
names = sorted({p.split('/')[-1].replace('_trajectory.npz', '') for p in _g.glob(f'{A}/episode_{EP:06d}/*_trajectory.npz')})
# kit slot order = order in the bundle npz
d = np.load(f'{RR}/selected20_final_v8_g10b/episode_{EP:06d}/object_poses.npz'); order = [str(n) for n in d['names']]
names = [n for n in order if n in names]
aT, am = load_target(A, EP, names); bT, bm = load_target(B, EP, names)
S = Scorer(EP, names, am, bm)
nf = min(min(len(t) for t in aT), min(len(t) for t in bT))
ach = np.stack([S.q(t[:nf]) for t in aT], 1)[:, None]; ref = np.stack([S.q(t[:nf]) for t in bT], 1)[:, None]
for slot in range(len(names)):
    ca, R = S.regs[slot]
    ach[:, :, slot] = M._convert_object_body(ach[:, :, slot], ca, R)
    ref[:, :, slot, :3] += np.einsum('...ij,j->...i', M._rotation_matrix(ref[:, :, slot, 3:]), S.anchor[slot])
al = M._align_registered_scene(ach, ref, S.axes)[:, 0]; rf = ref[:, 0]
ori = np.degrees(M._symmetry_orientation_errors(al[:, None], rf[:, None], S.axes, S.orders))[:, 0]
print('names', names, 'nf', nf)
for i, n in enumerate(names):
    e = (al[:, i, :3] - rf[:, i, :3]) * 100
    print(n, 'pos mean %.2f ori mean %.1f' % (np.linalg.norm(e, axis=1).mean(), ori[:, i].mean()), 'mean vec', e.mean(0).round(2))
    for f in list(range(0, nf, ev)) + [nf - 1]:
        print('  f%3d' % f, e[f].round(2), '%.2f' % np.linalg.norm(e[f]), 'ori %.1f' % ori[f, i])
