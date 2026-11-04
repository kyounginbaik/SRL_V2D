import sys, json, numpy as np, mujoco
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamF_margins/ep19/code')
from common19F import *
from scipy.spatial import cKDTree
G = np.array(json.load(open('/data1/SRL_V2D/challenge_reconstruction/recon_v15_20261022/episode_000019/change.json'))['G'])
man = json.loads((EPDIR / 'input_manifest.json').read_text()); m = mujoco.MjModel.from_xml_path(man['physical_scene'])
b = m.body('mini_sweeper').id; pts = []
for g in range(m.ngeom):
    if m.geom_bodyid[g] != b or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH: continue
    mid = m.geom_dataid[g]; v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
    Rg = np.zeros(9); mujoco.mju_quat2Mat(Rg, m.geom_quat[g]); pts.append(v @ Rg.reshape(3, 3).T + m.geom_pos[g]); print(m.geom(g).name, m.geom_contype[g], m.geom_conaffinity[g], len(v))
P = np.concatenate(pts); Pg = P @ G[:3, :3].T + G[:3, 3]
d1 = cKDTree(P).query(Pg)[0]; print('collision verts', len(P), 'G-image to orig: mean %.2f mm p95 %.2f max %.2f' % (d1.mean() * 1e3, np.percentile(d1, 95) * 1e3, d1.max() * 1e3))
print('mass', m.body_mass[b], 'ipos', m.body_ipos[b], 'G ipos', G[:3, :3] @ m.body_ipos[b] + G[:3, 3])
pk = packed_meshes()['mini_sweeper'][0]; pk = np.asarray(pk); d2 = cKDTree(pk).query(pk @ G[:3, :3].T + G[:3, 3])[0]; print('packed mesh G-image mean %.2f mm max %.2f' % (d2.mean() * 1e3, d2.max() * 1e3))
