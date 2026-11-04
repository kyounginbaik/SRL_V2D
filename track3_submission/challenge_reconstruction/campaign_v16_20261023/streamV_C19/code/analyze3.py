"""Support-plane consistency of the frame-0 poses: ours (setup world) vs compose."""
import json, numpy as np, sys, trimesh
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamV_C19/code')
from common19 import *
packed = packed_meshes(); reg = registration(packed); comp = compose_tracks()
man = json.loads((EPDIR / 'input_manifest.json').read_text()); S = np.array(man['setup_from_source'])
d = np.load(SRC / 'object_poses.npz'); W = d['world_from_object'].astype(float)
ours = {n: np.einsum('ij,tjk->tik', S, W[:, i]) for i, n in enumerate(NAMES)}; cb = {n: comp[n] @ reg[n]['M'] for n in NAMES}
V = {n: np.asarray(packed[n][0]) for n in NAMES}
def pts(T, n): return V[n] @ T[:3, :3].T + T[:3, 3]
for t in (0, 5, 10, 20, 230, 234):
    print('ours t', t, {n: round(float(pts(ours[n][t], n)[:, 2].min() * 100), 2) for n in NAMES})
# best support normal per object in compose world (frame 0): hull face normals; pick the face whose normal is closest to the
# direction "down" defined by the dust pan bottom face mapped from ours
def support_face(P, down):
    h = trimesh.convex.convex_hull(P); fn = h.face_normals; best = np.argmax(fn @ down); return fn[best], h
for frame in (0, 10, 20):
    for label, poses, down in (('ours', ours, np.array([0, 0, -1.])),):
        pass
    # gravity in compose world: map our setup -z through the rigid fit of the dust pan f0
    X = ours['mini_dust_pan'][frame] @ np.linalg.inv(cb['mini_dust_pan'][frame]); down_c = X[:3, :3].T @ np.array([0, 0, -1.])
    Xs = ours['mini_sweeper'][frame] @ np.linalg.inv(cb['mini_sweeper'][frame]); down_s = Xs[:3, :3].T @ np.array([0, 0, -1.])
    print('frame', frame, 'angle between "down" implied by dust pan vs sweeper (compose->ours)', round(float(np.degrees(np.arccos(np.clip(down_c @ down_s, -1, 1)))), 1))
    for n in NAMES:
        P = pts(cb[n][frame], n); Pd = P @ down_c; lo = Pd.max(); print('  compose', n, 'support depth along dustpan-down: spread of lowest 3%%: %.2f cm' % ((lo - np.percentile(Pd, 97)) * 100), 'lowest', round(lo * 100, 2))
    for n in NAMES:
        P = pts(ours[n][frame], n); print('  ours', n, 'lowest z', round(P[:, 2].min() * 100, 2))
# compose: lowest points of both objects along compose-implied gravity using the camera/fit of the dust pan
