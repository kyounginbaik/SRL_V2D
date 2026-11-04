import sys, json, numpy as np, mujoco
from scipy.spatial.transform import Rotation as R
from pathlib import Path
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamF_margins/ep19/code')
from common19F import EPDIR
G = np.array(json.loads(Path('/data1/SRL_V2D/challenge_reconstruction/recon_v15_20261022/episode_000019/change.json').read_text())['G'])
man = json.loads((EPDIR / 'input_manifest.json').read_text()); m = mujoco.MjModel.from_xml_path(man['physical_scene']); d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0); mujoco.mj_forward(m, d)
b = m.body('mini_sweeper').id; Rb = d.xmat[b].reshape(3, 3)
rv = R.from_matrix(G[:3, :3]).as_rotvec(); ax = rv / np.linalg.norm(rv); print('G angle', np.degrees(np.linalg.norm(rv)), 'axis body', ax.round(3), 'axis world', (Rb @ ax).round(3))
pts = []
for g in range(m.ngeom):
    if m.geom_bodyid[g] != b or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or not m.geom_contype[g]: continue
    mid = m.geom_dataid[g]; v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
    gm = np.zeros(9); mujoco.mju_quat2Mat(gm, m.geom_quat[g]); pts.append(v @ gm.reshape(3, 3).T + m.geom_pos[g])
P = np.concatenate(pts); c = P.mean(0); w, U = np.linalg.eigh((P - c).T @ (P - c) / len(P)); print('body extents (along PCA)', (np.sqrt(w) * 100).round(2), 'PCA axes body\n', U.round(3))
print('body bounds cm', (P.min(0) * 100).round(1), (P.max(0) * 100).round(1)); print('world body axes (cols = x,y,z)\n', Rb.round(3))
W = P @ Rb.T + d.xpos[b]; print('world z range cm', (W[:, 2].min() * 100).round(2), (W[:, 2].max() * 100).round(2))
