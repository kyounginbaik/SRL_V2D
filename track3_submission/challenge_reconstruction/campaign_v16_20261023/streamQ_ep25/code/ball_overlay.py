"""Project each board's mocap-ball vertices (mesh points protruding > 8 mm beyond the face planes) at a reference pose into the
Compose RGB frame (OpenCV-undistorted ego_cam_a, K of camera_motion) and draw them over the video frame; also the twins
(half turn about the long axis / about the mid axis through the slab centre). usage: ball_overlay.py POSES.npz FRAME OUT.jpg"""
import sys, numpy as np, cv2, trimesh
from scipy.spatial.transform import Rotation as R
SRC = '/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/collision_recon_slots_v3/episode_000025'
V = '/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/physics_slots_v3/rgb/episode_000025/anycalib/undistorted.mp4'
K = np.array([[1149.804931640625, 0, 1013.5], [0, 1149.730712890625, 759.5], [0, 0, 1]])
poses, k, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
T = np.load(poses)['world_from_object']; Wc = np.load(SRC + '/object_poses.npz')['world_from_rgb'].astype(float)
cap = cv2.VideoCapture(V); cap.set(cv2.CAP_PROP_POS_FRAMES, k); ok, im = cap.read(); assert ok
cols = {'orig': (0, 0, 255), 'twin_long': (255, 0, 0), 'twin_mid': (0, 255, 0)}
for i, n in enumerate(['wooden_piece_1', 'wooden_piece_2']):
    v = np.asarray(trimesh.load(f'{SRC}/assets/{n}_visual.obj', force='mesh').vertices); c0 = v.mean(0); w, U = np.linalg.eigh((v - c0).T @ (v - c0))
    q = (v - c0) @ U; h, e = np.histogram(q[:, 0], bins=80); pk = np.sort(e[np.argsort(h)[-2:]] + np.diff(e)[0] / 2)
    balls = v[(q[:, 0] > pk[1] + .008) | (q[:, 0] < pk[0] - .008)]
    sc = c0 + U @ np.array([pk.mean(), 0, 0])
    outline = c0 + np.array([[pk.mean(), a, b] for a, b in [(-.062, -.105), (-.062, .105), (.062, .105), (.062, -.105)]]) @ U.T
    for nm, Rb in [('orig', np.eye(3)), ('twin_long', R.from_rotvec(np.pi * U[:, 2]).as_matrix()), ('twin_mid', R.from_rotvec(np.pi * U[:, 1]).as_matrix())]:
        Tk = T[k, i].copy(); Rn = Tk[:3, :3] @ Rb; tn = Tk[:3, :3] @ sc + Tk[:3, 3] - Rn @ sc
        for pts, rad in [(balls, 3), (outline, 0)]:
            P = pts @ Rn.T + tn; C = (np.linalg.inv(Wc[k]) @ np.c_[P, np.ones(len(P))].T).T[:, :3]; uv = (C @ K.T); uv = uv[:, :2] / uv[:, 2:]
            if rad: [cv2.circle(im, (int(a), int(b)), rad, cols[nm], -1) for a, b in uv[::7]]
            elif nm == 'orig': cv2.polylines(im, [uv.astype(np.int32)], True, (0, 255, 255), 2)
    cv2.putText(im, 'red=ref balls blue=twin(long) green=twin(mid) yellow=ref outline', (30, 60), 0, 1.5, (255, 255, 255), 3)
cv2.imwrite(out, cv2.resize(im, (1014, 760)))
