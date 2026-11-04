"""Rest geometry of ours vs compose-derived poses in the streamB level setup frame: object up-axis tilt vs +z, bottom heights,
fitted support planes. usage: rest_geometry.py REFERENCE_NPZ"""
import sys, json, numpy as np, trimesh
from pathlib import Path
from scipy.spatial.transform import Rotation as R
CR = Path('/data1/SRL_V2D/challenge_reconstruction'); SRC = CR / 'final_recon/episode_000008'
S = np.array(json.load(open(CR / 'campaign_v16_20261023/streamB_tilt38/ep8/calibrated_inputs/episodes/episode_000008/input_manifest.json'))['setup_from_source'])
ours = np.load(SRC / 'object_poses.npz')['world_from_object'].astype(float); new = np.load(sys.argv[1])['world_from_object'].astype(float)
names = ['beige_cup', 'white_pot']; V = {n: np.asarray(trimesh.load(SRC / f'assets/{n}.obj', force='mesh').vertices) for n in names}
# object 'up' axis in body frame = direction that is most +z in OUR level-frame rest pose at frame 0 (lowest-point support normal)
for k, n in enumerate(names):
    up_body = (S[:3, :3] @ ours[0, k, :3, :3]).T @ np.array([0, 0, 1.])
    for lab, P in (('ours', ours), ('compose', new)):
        Ts = np.einsum('ij,tjk->tik', S, P[:, k]); up = Ts[:, :3, :3] @ up_body; tilt = np.degrees(np.arccos(np.clip(up[:, 2], -1, 1)))
        pts = np.einsum('tij,vj->tvi', Ts[:, :3, :3], V[n]) + Ts[:, None, :3, 3]; z = pts[..., 2].min(1); i = pts[..., 2].argmin(1)
        low = pts[np.arange(len(pts)), i]
        print(n, lab, 'f0 tilt %.1f deg dir(xy) %s bottom %.2f cm lowest pt %s | last tilt %.1f bottom %.2f | centroid f0 %s' % (tilt[0], np.round(up[0, :2], 3), z[0] * 100, np.round(low[0] * 100, 1), tilt[-1], z[-1] * 100, np.round(Ts[0, :3, 3] * 100, 1)))
