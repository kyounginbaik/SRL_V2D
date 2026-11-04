"""Project the physics collision geometry of each board (densely sampled surface points of the slot-cut convex parts) at a reference
pose onto the Compose RGB frame; the slot appears as an empty strip. usage: slot_overlay.py POSES.npz FRAME OUT.jpg [COLL_DIR_1 COLL_DIR_2]"""
import sys, glob, numpy as np, cv2, trimesh
SRC = '/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/collision_recon_slots_v3/episode_000025'
V = '/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/physics_slots_v3/rgb/episode_000025/anycalib/undistorted.mp4'
K = np.array([[1149.804931640625, 0, 1013.5], [0, 1149.730712890625, 759.5], [0, 0, 1]])
C = '/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/ep25_v15/collision/'
poses, k, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]; dirs = sys.argv[4:6] if len(sys.argv) > 5 else [C + 'wooden_piece_1_cut3mm', C + 'wooden_piece_2_cut3mm']
T = np.load(poses)['world_from_object']; Wc = np.load(SRC + '/object_poses.npz')['world_from_rgb'].astype(float)
cap = cv2.VideoCapture(V); cap.set(cv2.CAP_PROP_POS_FRAMES, k); ok, im = cap.read()
ov = im.copy()
for i, d in enumerate(dirs):
    m = trimesh.util.concatenate([trimesh.load(f, force='mesh') for f in sorted(glob.glob(d + '/part_*.obj'))]); P = trimesh.sample.sample_surface(m, 60000, seed=0)[0]
    P = P @ T[k, i, :3, :3].T + T[k, i, :3, 3]; c = (np.linalg.inv(Wc[k]) @ np.c_[P, np.ones(len(P))].T).T[:, :3]; uv = c @ K.T; uv = (uv[:, :2] / uv[:, 2:]).astype(int)
    for a, b in uv: cv2.circle(ov, (a, b), 1, (255, 0, 255) if i else (0, 255, 255), -1)
im = cv2.addWeighted(im, 0.55, ov, 0.45, 0)
cv2.putText(im, f'f{k} collision points: yellow=board1 magenta=board2 (slot = empty strip)', (30, 60), 0, 1.4, (255, 255, 255), 3)
cv2.imwrite(out, im[300:1400, 300:1900])
