"""K33-inhand (GT-free): spoon pose relative to OUR HaWoR wrist (hand_motion_prior_v2, gravity-aligned world) for a bundle. During a rigid
grasp wrist_from_spoon should be ~constant; report per frame the deviation (cm of COM, deg rotation, deg about the spoon long axis)
from the segment median. usage: grasp33.py BUNDLE_EP_DIR out.txt"""
import sys, json, numpy as np, trimesh
from scipy.spatial.transform import Rotation as Rot
B = sys.argv[1]
C = '/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/canonical/evaluation/episode_000033'
z = np.load(f'{B}/object_poses.npz', allow_pickle=False); S = list(z['names']).index('wooden_spoon'); Tw = z['world_from_object'][:, S]
hp = np.load(f'{C}/hand_motion_prior_v2.npz', allow_pickle=False); HS = list(hp['names']).index('wooden_spoon')
js = json.load(open(f'{C}/hand_motion_prior_v2.json'))['objects']['wooden_spoon']['segments']
m = trimesh.load(f'{B}/meshes/wooden_spoon.obj', force='mesh'); v = np.asarray(m.vertices); com = np.asarray(m.center_mass)
c0 = v - v.mean(0); w, U = np.linalg.eigh(c0.T @ c0 / len(c0)); long = U[:, 2]
if long[np.argmax(np.abs(long))] < 0: long = -long
# the prior and the bundle may live in different world frames: use the bundle world (same 'world' as world_from_wrist only if equal)
print('segments', [(s['hand'], s['start'], s['stop']) for s in js])
out = []
for s in js:
    ts = [t for t in range(s['start'], s['stop']) if hp['wrist_valid'][t, HS]]
    Wr = hp['world_from_wrist'][ts, HS]; rel = np.array([np.linalg.inv(Wr[i]) @ Tw[t] for i, t in enumerate(ts)])
    Rm = Rot.from_matrix(rel[:, :3, :3]).mean().as_matrix(); pm = np.median(rel[:, :3, :3] @ com + rel[:, :3, 3], 0)
    for i, t in enumerate(ts):
        d = Rot.from_matrix(Rm.T @ rel[i, :3, :3]).as_rotvec(); p = rel[i, :3, :3] @ com + rel[i, :3, 3]
        out.append(f"f{t:3d} seg{s['start']}-{s['stop']} dCOM {np.linalg.norm(p - pm) * 100:5.1f}cm  dRot {np.degrees(np.linalg.norm(d)):5.1f}d  roll {np.degrees(d @ long):6.1f}d stereo={int(hp['wrist_stereo_supported'][t, HS])}")
open(sys.argv[2], 'w').write('\n'.join(out) + '\n'); print('\n'.join(out[::4]))
