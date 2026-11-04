"""ep-level: f0 relative-placement discrepancy (other recon vs ours) expressed in OUR camera-0 frame, vs the viewing ray of slot 0.
usage: rvr_cam.py EP OURS_TARGET OTHER_TARGET"""
import sys, numpy as np
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG10_mp/code')
from g10lib import *
EP, A, B = int(sys.argv[1]), sys.argv[2], sys.argv[3]
d = np.load(f'{RR}/selected20_final_v8_g10b/episode_{EP:06d}/object_poses.npz'); names = [str(n) for n in d['names']]
aT, am = load_target(A, EP, names); bT, bm = load_target(B, EP, names)
S = Scorer(EP, names, am, bm)
nf = min(min(len(t) for t in aT), min(len(t) for t in bT))
ach = np.stack([S.q(t[:nf]) for t in aT], 1)[:, None]; ref = np.stack([S.q(t[:nf]) for t in bT], 1)[:, None]
for slot in range(len(names)):
    ca, R = S.regs[slot]
    ach[:, :, slot] = M._convert_object_body(ach[:, :, slot], ca, R)
    ref[:, :, slot, :3] += np.einsum('...ij,j->...i', M._rotation_matrix(ref[:, :, slot, 3:]), S.anchor[slot])
al = M._align_registered_scene(ach, ref, S.axes)[:, 0]; rf = ref[:, 0]; ac = ach[:, 0]
# alignment rotation: al = ac @ Rot.T + off
X = ac[:, :, :3].reshape(-1, 3); Y = al[:, :, :3].reshape(-1, 3)
xc, yc = X.mean(0), Y.mean(0); u, s, vt = np.linalg.svd((X - xc).T @ (Y - yc)); Rk = (u @ vt).T
if np.linalg.det(Rk) < 0: print('reflection?')
cam0 = d['world_from_rgb'][0]
for i, n in enumerate(names):
    e = al[:, i, :3] - rf[:, i, :3]          # in other's frame
    e_ours = e @ Rk                            # rotate back to our world
    e_cam = (e_ours) @ cam0[:3, :3]            # our world -> camera0 (rotation part)
    pass
    for f in [0, 20, 50, 80, 100, 120, 140, 160, nf - 1]:
        camf = d["world_from_rgb"][f]; pc = np.linalg.inv(camf) @ np.r_[ac[f, i, :3], 1]; ray = pc[:3] / np.linalg.norm(pc[:3]); ec = e_ours[f] @ camf[:3, :3]
        print(n, "f", f, "err camf cm", (ec * 100).round(2), "|e| %.2f along own ray %.2f" % (np.linalg.norm(ec) * 100, float(ec @ ray * 100)), "dist %.1f" % (np.linalg.norm(pc[:3]) * 100))
p0 = np.linalg.inv(cam0) @ np.r_[ac[0, 0, :3], 1]; ray0 = p0[:3] / np.linalg.norm(p0[:3])
print('slot0 (%s) f0 ray cam0' % names[0], ray0.round(3), 'dist cm', round(float(np.linalg.norm(p0[:3]) * 100), 1))
