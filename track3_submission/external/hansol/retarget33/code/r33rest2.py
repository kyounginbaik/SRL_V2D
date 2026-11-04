"""R33: do the cup and spoon rest on ONE common plane at f0 / f347 (GT-free)? Spoon rest plane = the convex-hull facet of the spoon whose
outward normal points most downward; cup rest plane = its bottom (symmetry axis). Reports each object's rest normal vs the measured table
normal, the angle between the two objects' rest normals (independent of the table), and that relative error split into spoon roll (about
its long axis) and pitch."""
import numpy as np, mujoco, trimesh
from pathlib import Path
exec(open('/data1/hansol/retarget33/code/r33rest.py').read().split('for lab in')[0])
def deg(a, b): return np.degrees(np.arccos(np.clip(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b), -1, 1)))
import sys
for lab in (sys.argv[1:] or ['R33_HGA7b', 'R33_HGA_r6']):
    W = R / 'worlds' / lab / 'ep33/episode_corrected/episode_000033'; m = mujoco.MjModel.from_xml_path(str(W / 'stage2/scene.xml')); d = mujoco.MjData(m)
    Q = np.load(W / 'reference_trajectory.npz', allow_pickle=True)['qpos']; print(f'\n=== {lab}')
    for f in (0, 10, 20, 330, 347):
        d.qpos[:] = Q[f]; mujoco.mj_kinematics(m, d)
        X = body_verts(m, d, 'blue_cup'); c = X.mean(0); w, U = np.linalg.eigh(np.cov((X - c).T)); k = 0 if abs(np.log(w[1] / w[0])) > abs(np.log(w[2] / w[1])) else 2
        ax = U[:, k]; h = (X - c) @ ax; r = np.linalg.norm((X - c) - np.outer(h, ax), axis=1); near = r < 0.5 * np.percentile(r, 95); lo, hi = h.min(), h.max()
        if not near[h < lo + 0.15 * (hi - lo)].mean() > near[h > hi - 0.15 * (hi - lo)].mean(): ax = -ax   # ax: bottom -> open end, i.e. "up" if standing
        S = body_verts(m, d, 'wooden_spoon'); hull = trimesh.convex.convex_hull(S); fn = hull.face_normals; fa = hull.area_faces
        i = np.argmin(fn[:, 2] - 1e-9 * fa); up_s = -fn[i]
        # merge near-coplanar downward facets (within 3 deg) weighted by area for a stable rest normal
        sel = np.degrees(np.arccos(np.clip(fn @ fn[i], -1, 1))) < 3; up_s = -(fn[sel] * fa[sel, None]).sum(0); up_s /= np.linalg.norm(up_s)
        cS = S.mean(0); w2, U2 = np.linalg.eigh(np.cov((S - cS).T)); L = U2[:, 2]
        rel = deg(ax, up_s); e = np.cross(up_s, ax); roll = np.degrees(np.arcsin(np.clip(e @ L, -1, 1)))
        print(f'  f{f:3d}: cup up vs table normal {deg(ax, [0,0,1]):5.1f} deg | spoon rest normal vs table normal {deg(up_s, [0,0,1]):5.1f} deg (rest facet area {100*100*fa[sel].sum():.1f} cm2) |'
              f' cup up vs spoon rest normal {rel:5.1f} deg, of which about the spoon long axis (roll) {roll:+5.1f} deg')
