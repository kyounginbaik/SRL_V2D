"""decompose orientation error of a symmetric object into axis tilt and spin about the axis. usage: axis_decomp.py TL_NPZ EP OBJ SLOT [BIN]"""
import sys, numpy as np, trimesh
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, '/data1/SRL_V2D/challenge_submissions_SRL/analysis_20261020/scripts'); import public_kit_score as K
z = np.load(sys.argv[1]); ep = int(sys.argv[2]); n = sys.argv[3]; s = int(sys.argv[4]); B = int(sys.argv[5]) if len(sys.argv) > 5 else 15
m = trimesh.load(f'/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/target_final_v5/episode_{ep:06d}/{n}_mesh_metric.obj', force='mesh'); m.merge_vertices()
ax, o = K.symmetry_axis(n, np.asarray(m.vertices)); print('axis', ax, 'order', o, 'extent', np.ptp(m.vertices, 0))
qa = R.from_quat(z['ach'][:, 0, s, 3:]); qr = R.from_quat(z['ref'][:, 0, s, 3:])
aa = qa.apply(ax); ar = qr.apply(ax); tilt = np.degrees(np.arccos(np.clip((aa * ar).sum(1), -1, 1)))
d = (qr.inv() * qa)  # body-frame relative rotation
rv = d.as_rotvec(); spin = np.degrees(rv @ ax); tot = np.degrees(d.magnitude())
# world tilt of axis vs world z of each
for i in range(0, len(tilt), B):
    sl = slice(i, i + B); print(f'f{i:3d} tilt {tilt[sl].mean():5.1f} spin {spin[sl].mean():+6.1f} tot {tot[sl].mean():5.1f} | ach-axis-up {np.degrees(np.arccos(np.clip(aa[sl,2],-1,1))).mean():5.1f} ref-axis-up {np.degrees(np.arccos(np.clip(ar[sl,2],-1,1))).mean():5.1f}')
