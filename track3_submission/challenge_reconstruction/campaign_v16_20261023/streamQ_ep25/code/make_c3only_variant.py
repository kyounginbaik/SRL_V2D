"""Evidence helper (not a correction): board 2 := ORIGINAL board 1 x the geometric assembly transform (C3 of correct_ep25_v2.py)
for frames 230-355 (blend 220-230), board 1 and everything else original, no table resting. Lets the camera-frame stereo test judge
the relative-pose part of C3 independently of the table-offset part. usage: make_c3only_variant.py OUT.npz"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); import correct_ep25_v2 as C
src = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/collision_recon_slots_v3/episode_000025')
z = dict(np.load(src / 'object_poses.npz')); T0 = z['world_from_object']; T = T0.copy(); F = C.frames(src)
for k in range(C.C3[0], C.C3[2] + 1):
    X, _ = C.assembly(F, T0[k, 0], T0[k, 1]); Tn = T0[k, 0] @ X; w = 1. if k >= C.C3[1] else C.smooth((k - C.C3[0]) / (C.C3[1] - C.C3[0])); T[k, 1] = C.blend_pose(T0[k, 1], Tn, w)
z['world_from_object'] = T; np.savez(sys.argv[1], **z)
