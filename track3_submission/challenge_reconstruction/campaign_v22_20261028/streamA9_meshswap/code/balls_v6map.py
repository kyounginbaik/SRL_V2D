"""A9 GT-free marker-side test with OUR validated tracking: ball detections -> v6 body frame (final_v6 camera_from_object)
-> submitted body frame via T_icp. For each detection mode: distance to the submitted mesh vs its own self-twin, and to the
v6 mesh (expressed in the same frame) vs its twin. A mode 'votes' for a mesh if it is within TOL of it and > TOL from the twin.
Usage: balls_v6map.py OUT.json EP:OBJ ..."""
import sys, json, numpy as np
from scipy.spatial import cKDTree
from audit import *
TOL = 3.0
def votes(B, mesh_pts, H):
    t0, t1 = cKDTree(mesh_pts), cKDTree(apply(H, mesh_pts)); out = []
    for c, k in modes(B):
        a, b = t0.query(c)[0] * 1e3, t1.query(c)[0] * 1e3
        out.append(dict(c_cm=(c * 100).round(2).tolist(), n=k, d_mesh_mm=round(a, 2), d_twin_mm=round(b, 2), vote='mesh' if a < TOL <= b else 'twin' if b < TOL <= a else 'both' if a < TOL and b < TOL else 'none'))
    s = lambda v: sum(x['n'] for x in out if x['vote'] == v)
    return out, dict(mesh=s('mesh'), twin=s('twin'), both=s('both'), none=s('none'))
res = []
for spec in sys.argv[2:]:
    ep, obj = int(spec.split(':')[0]), spec.split(':')[1]
    r = next(x for x in json.loads((CAMP / f'streamA9_meshswap/out/audit_ep{ep}.json').read_text()) if x['obj'] == obj)
    n6, z6 = v6_poses(ep); C6 = z6['camera_from_object'][:, n6.index(obj)]; Ti = np.array(r['Ticp'])
    Ms = load(comp_component(ep) / 'meshes' / ename(ep) / f'{obj}.obj'); M6 = load(V6 / ename(ep) / 'meshes' / f'{obj}.obj')
    Ps = samp(Ms); P6 = apply(Ti, samp(M6)); B = apply(Ti, balls_body(ep, obj, C6))
    _, Hs, _ = self_twin(Ps, cKDTree(Ps)); _, H6, _ = self_twin(P6, cKDTree(P6))
    ms, vs = votes(B, Ps, Hs); m6, v6v = votes(B, P6, H6)
    res.append(dict(ep=ep, obj=obj, sub_votes=vs, v6_votes=v6v, sub_modes=ms, v6_modes=m6))
    print(ep, obj, 'submitted mesh votes (detections in modes)', vs, '| v6 mesh votes', v6v, flush=True)
    for a, b in zip(ms, m6): print('   mode', a['c_cm'], 'n', a['n'], 'sub', a['d_mesh_mm'], '/', a['d_twin_mm'], a['vote'], '| v6', b['d_mesh_mm'], '/', b['d_twin_mm'], b['vote'])
Path(sys.argv[1]).write_text(json.dumps(res, indent=1, default=float) + '\n')
