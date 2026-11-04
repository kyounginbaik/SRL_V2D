"""A9 task 2: MESH-ONLY swap. Copies the v23b legacy component (never overwrites) and replaces ONLY
meshes/<episode>/<obj>.obj with the final_v6 mesh expressed in the component's body frame (x_sub = T_icp x_v6, T_icp from audit.py:
ICP of the v6 mesh onto the submitted mesh, started at the pose-derived T_cam, i.e. in the basin that keeps the clip's marker convention).
Poses, policy, actions, the portable bundle (scene, collision parts, source meshes) are untouched.
Usage: stage.py TAG EP OBJ [OBJ ...]"""
import sys, json, shutil, hashlib, numpy as np
from pathlib import Path
from a9lib import *
tag, ep, objs = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
src = comp_component(ep); dst = CAMP / 'streamA9_meshswap/staging' / tag / 'components' / ename(ep)
assert not dst.exists(), dst
shutil.copytree(src, dst, symlinks=True)
A = {r['obj']: r for r in json.loads((CAMP / f'streamA9_meshswap/out/audit_ep{ep}.json').read_text())}
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
rec = dict(stream='A9', tag=tag, episode=ename(ep), source_component=str(src), swapped={}, unchanged={})
read = json.loads((dst / 'readiness.json').read_text())
for obj in objs:
    T = np.array(A[obj]['Ticp']); v6 = V6 / ename(ep) / 'meshes' / f'{obj}.obj'; out = dst / 'meshes' / ename(ep) / f'{obj}.obj'; old_sha = sha(out)
    lines = []
    for ln in v6.read_text().splitlines():
        if ln.startswith('v '):
            p = ln.split(); x = np.array([float(p[1]), float(p[2]), float(p[3])]); y = T[:3, :3] @ x + T[:3, 3]
            lines.append('v %.9f %.9f %.9f' % tuple(y) + ('' if len(p) == 4 else ' ' + ' '.join(p[4:])))
        elif ln.startswith('vn '):
            p = ln.split(); y = T[:3, :3] @ np.array([float(p[1]), float(p[2]), float(p[3])]); lines.append('vn %.6f %.6f %.6f' % tuple(y))
        else: lines.append(ln)
    out.write_text('\n'.join(lines) + '\n'); np.save(dst / 'meshes' / ename(ep) / f'{obj}.a9_T_sub_from_v6.npy', T)
    rec['swapped'][obj] = dict(old_sha256=old_sha, old_md5=A[obj]['sub_md5'], v6_source=str(v6), v6_sha256=sha(v6), new_sha256=sha(out),
                               T_sub_from_v6=T.tolist(), T_cam_pose_derived=A[obj]['Tcam'], icp_vs_cam_deg=A[obj]['icp_vs_cam_deg'],
                               mean_sym_surface_dist_to_old_cm=A[obj]['symdist_icp_cm'], twin_basin_mean_sym_dist_cm=A[obj].get('twin_icp_symdist_cm'))
    if obj in read.get('meshes', {}):
        read['meshes'][obj].update(sha256=sha(out), a9_mesh_swap=dict(previous_sha256=old_sha, source=str(v6), transform='x_sub = T x_v6, see a9_meshswap.json'))
for p in sorted((dst / 'meshes' / ename(ep)).glob('*.obj')):
    if p.stem not in objs: rec['unchanged'][p.stem] = sha(p)
(dst / 'readiness.json').write_text(json.dumps(read, indent=2) + '\n')
(dst / 'a9_meshswap.json').write_text(json.dumps(rec, indent=1) + '\n')
# the stray .npy must not be picked up as a mesh: move it next to the record
for f in (dst / 'meshes' / ename(ep)).glob('*.npy'): shutil.move(str(f), dst / f.name)
print(json.dumps({k: (v if k != 'swapped' else {o: {kk: vv for kk, vv in d.items() if 'T_' not in kk} for o, d in v.items()}) for k, v in rec.items()}, indent=1))
