"""GT-free comparison of two recon bundles for one episode, kit-style frame-0 alignment (no registration: same meshes).
Translation pivot = slot 0 at f0, rotation = anchor slot at f0. Cup is treated as axis-symmetric (tilt only), others full rotation.
Also prints the anchor sensitivity: induced time-mean position error per 1 deg of anchor f0 rotation error (worst axis), per object.
usage: cmp_bundles.py EP BUNDLE_A BUNDLE_B ANCHOR_SLOT"""
import sys, json, numpy as np
from scipy.spatial.transform import Rotation as R
ep, A, B, anc = int(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4])
E = f'episode_{ep:06d}'
a = np.load(f'{A}/{E}/object_poses.npz', allow_pickle=True); b = np.load(f'{B}/{E}/object_poses.npz', allow_pickle=True)
print('keys A', {k: a[k].shape for k in a.files}); print('keys B', {k: b[k].shape for k in b.files})
names = [str(n) for n in a['names']]
Wa, Wb = a['world_from_object'], b['world_from_object']
n = min(len(Wa), len(Wb))
SYM = {'blue_cup', 'beige_cup'}


def align(Wref, Wx):
    # rigid T so that T @ Wx[0, anc] rotation == Wref[0, anc] rotation, and slot-0 f0 positions coincide
    Rr = Wref[0, anc, :3, :3] @ Wx[0, anc, :3, :3].T
    t = Wref[0, 0, :3, 3] - Rr @ Wx[0, 0, :3, 3]
    T = np.eye(4); T[:3, :3] = Rr; T[:3, 3] = t
    return np.einsum('ij,fojk->foik', T, Wx)


def err(Wref, Wx):
    Al = align(Wref, Wx); out = {}
    for s, nm in enumerate(names):
        dp = np.linalg.norm(Al[:n, s, :3, 3] - Wref[:n, s, :3, 3], axis=1) * 100
        if nm in SYM:
            za = Al[:n, s, :3, 2]; zb = Wref[:n, s, :3, 2]
            ang = np.degrees(np.arccos(np.clip(np.einsum('fi,fi->f', za, zb), -1, 1)))
        else:
            Rd = np.einsum('fji,fjk->fik', Al[:n, s, :3, :3], Wref[:n, s, :3, :3])
            ang = np.degrees(np.arccos(np.clip((np.trace(Rd, axis1=1, axis2=2) - 1) / 2, -1, 1)))
        out[nm] = (round(float(dp.mean()), 2), round(float(ang.mean()), 2), round(float(dp.max()), 2))
    return out


print('names', names, 'frames', len(Wa), len(Wb), 'anchor slot', anc)
print('A vs B (B as reference): mean cm, mean deg, max cm', err(Wb, Wa))
# anchor sensitivity: perturb A's anchor f0 by 1 deg about world x/y/z -> the alignment rotates the whole scene about slot 0
for ax in 'xyz':
    P = Wa.copy(); dR = R.from_euler(ax, 1, degrees=True).as_matrix()
    P[0, anc, :3, :3] = dR @ P[0, anc, :3, :3]
    print(f'1 deg anchor f0 error about world {ax}:', {k: v[0] for k, v in err(Wa, P).items()})
# lever: distance of each object from slot-0 f0 position over time
for s, nm in enumerate(names):
    d = np.linalg.norm(Wa[:n, s, :3, 3] - Wa[0, 0, :3, 3], axis=1) * 100
    print(f'lever {nm}: mean {d.mean():.1f} cm, max {d.max():.1f} cm; path length {np.linalg.norm(np.diff(Wa[:n, s, :3, 3], axis=0), axis=1).sum() * 100:.0f} cm')
