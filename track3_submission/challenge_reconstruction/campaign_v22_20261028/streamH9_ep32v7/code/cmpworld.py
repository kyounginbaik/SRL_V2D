"""B9: compare the v6 world (W6) with the v5 world (streamB W1): world transform delta, per-object reference shift (physics frame), frame-0 init change."""
import sys, json, numpy as np, trimesh
from pathlib import Path
from scipy.spatial.transform import Rotation as R
W1 = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamB_ep3233/ep32/W1'); W6 = Path(sys.argv[1]); EN = 'episode_000032'
r1 = json.loads((W1 / f'source_corrected/{EN}/target_retarget.json').read_text()); r6 = json.loads((W6 / f'source_corrected/{EN}/target_retarget.json').read_text())
X1, X6 = np.array(r1['world_transform_X']), np.array(r6['world_transform_X']); D = X6 @ np.linalg.inv(X1)
print('X6 X1^-1: rot deg', np.degrees(R.from_matrix(D[:3, :3]).magnitude()), 'rotvec', np.degrees(R.from_matrix(D[:3, :3]).as_rotvec()).round(3), 'trans cm', (D[:3, 3] * 100).round(3))
z1 = np.load(W1 / f'source_corrected/{EN}/object_poses.npz'); z6 = np.load(W6 / f'source_corrected/{EN}/object_poses.npz'); names = [str(x) for x in z1['names']]
SRC = Path('/data1/SRL_V2D/challenge_reconstruction/final_recon') / EN
out = {}
for n in ('blue_cup', 'wooden_spoon'):
    i = names.index(n); c = trimesh.load(SRC / f'assets/{n}.obj', force='mesh', process=False).centroid
    A, Bm = z1['world_from_object'][:, i], z6['world_from_object'][:, i]
    p = np.linalg.norm((A[:, :3, :3] @ c + A[:, :3, 3]) - (Bm[:, :3, :3] @ c + Bm[:, :3, 3]), axis=1) * 100
    d = np.degrees([R.from_matrix(x.T @ y).magnitude() for x, y in zip(A[:, :3, :3], Bm[:, :3, :3])])
    out[n] = dict(mean_cm=float(p.mean()), max_cm=float(p.max()), mean_deg=float(d.mean()), f0_cm=float(p[0]), f0_deg=float(d[0]), f1_cm=float(p[1]), f1_deg=float(d[1]))
print(json.dumps(out, indent=1))
