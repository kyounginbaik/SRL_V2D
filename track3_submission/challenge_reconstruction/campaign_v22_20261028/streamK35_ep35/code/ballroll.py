"""GT-free mesh-internal consistency: roll of our spoon shell about its long axis (through the shell centroid) that best seats the
pooled GT-free ball centres (G18 out/ballcenters.json; r 6.06 mm) ON the shell surface (signed distance ~ r)."""
import json, numpy as np, trimesh
from scipy.spatial.transform import Rotation as Rot
G18 = '/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/streamG18_spoonmesh'
m = trimesh.load('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/selected20_final_v14_g17/episode_000035/meshes/wooden_spoon.obj', force='mesh')
parts = m.split(only_watertight=False); outer = max(parts, key=lambda p: p.area); print('outer', len(outer.vertices), 'area cm2', round(outer.area * 1e4, 1))
bc = json.load(open(f'{G18}/out/ballcenters.json')); print(json.dumps(bc)[:600])
pe = bc['per_episode_centres']; r = bc['radius_mm_median'] / 1000
pool = {k: np.median([pe[e][k] for e in pe], 0) for k in ('near', 'far', 'tip')}
print('pooled centres', {k: v.round(4).tolist() for k, v in pool.items()})
v = np.asarray(outer.vertices); c0 = v.mean(0); cc = v - c0; w, U = np.linalg.eigh(cc.T @ cc / len(cc)); ax = U[:, 2]
if ax[0] < 0: ax = -ax
print('long axis', ax.round(3), 'centroid', c0.round(4))
from scipy.spatial import cKDTree
def sd(mesh, pts):   # + inside (trimesh convention), via dense surface samples + face normals
    P, fi = trimesh.sample.sample_surface_even(mesh, 300000, seed=0); N = mesh.face_normals[fi]
    d, i = cKDTree(P).query(pts); sgn = np.sign(np.einsum('ij,ij->i', pts - P[i], N[i])); return -sgn * d
rows = []
for deg in np.arange(-30, 31, 2.5):
    E = Rot.from_rotvec(ax * np.radians(deg)).as_matrix(); mm = outer.copy(); mm.vertices = (v - c0) @ E.T + c0
    out = {}
    for src, cen in [('pool', pool)] + [(e, {k: np.array(pe[e][k]) for k in ('near', 'far')}) for e in pe]:
        d = -sd(mm, np.array([cen['near'], cen['far']]))      # + outside
        out[src] = d
    allc = np.array([out[e] for e in pe]); cost = np.mean((allc - r) ** 2)
    rows.append((deg, out['pool'][0] * 1000, out['pool'][1] * 1000, np.sqrt(cost) * 1000))
for deg, a, b, c in rows: print(f'shell roll {deg:+6.1f}: pooled near {a:5.1f}mm far {b:5.1f}mm (ideal {r*1000:.1f}) | per-ep rms dev {c:5.2f}mm')
