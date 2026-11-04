"""G12 bundle builder (GT-free; ep32 spoon only). New bundle = copy of selected20_final_v9_g11; the ep32 wooden_spoon pose is
final_v6's spoon rolled about its long (PCA) axis through the mesh centroid (rollscan_ep32 axis/centroid, as G10) by the
G12 roll track (deg per frame, relative to final_v6). Meshes, the cup, cameras and all other episodes untouched. Never overwrites.
usage: build_v10.py OUT_BUNDLE ROLL_NPZ KEY [EP20_FROM=v9|v8]"""
import sys, json, shutil, numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as Rot
RR = Path('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026'); C = '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028'
OUT = Path(sys.argv[1]); RNPZ, KEY = sys.argv[2], sys.argv[3]; EP20 = sys.argv[4] if len(sys.argv) > 4 else 'v9'
assert not OUT.exists(), f'{OUT} exists (never overwrite)'
shutil.copytree(RR / 'selected20_final_v9_g11', OUT)
if EP20 == 'v8':
    shutil.copy2(RR / 'selected20_final_v8_g10b/episode_000020/object_poses.npz', OUT / 'episode_000020/object_poses.npz')
roll = np.load(RNPZ)[KEY].astype(float); roll[0] = 0.0
R = np.load(f'{C}/streamG10_mp/out/rollscan_ep32.npz'); c, ax = R['centroid'], R['axis']
d6 = np.load(RR / 'selected20_final_v6/episode_000032/object_poses.npz')
fol = OUT / 'episode_000032'; d = dict(np.load(fol / 'object_poses.npz')); names = [str(n) for n in d['names']]; s = names.index('wooden_spoon')
assert np.allclose(d['world_from_rgb'], d6['world_from_rgb'])
cam = d6['camera_from_object'][:, s].copy()
for t in range(len(cam)):
    if abs(roll[t]) < 1e-6: continue
    Rr = Rot.from_rotvec(np.radians(roll[t]) * ax).as_matrix(); X = np.eye(4); X[:3, :3] = Rr; X[:3, 3] = c - Rr @ c
    cam[t] = cam[t] @ X
d['camera_from_object'] = d['camera_from_object'].copy(); d['camera_from_object'][:, s] = cam
d['world_from_object'] = d['world_from_object'].copy(); d['world_from_object'][:, s] = d['world_from_rgb'] @ cam
np.savez(fol / 'object_poses.npz', **d)
info = dict(rule='ep32 spoon roll = G12 constrained ball Viterbi (streamG12/code/a32b.py) relative to final_v6', roll_npz=RNPZ, key=KEY,
            roll_deg=np.round(roll, 2).tolist(), ep20=EP20, base='selected20_final_v9_g11')
(fol / 'g12_spoon_roll.json').write_text(json.dumps(info, indent=1))
(OUT / 'G12_CHANGES.json').write_text(json.dumps(dict(base='selected20_final_v9_g11', changed={'32': 'wooden_spoon roll (G12)', '20': f'pitcher from {EP20}'}), indent=1))
print('built', OUT)
