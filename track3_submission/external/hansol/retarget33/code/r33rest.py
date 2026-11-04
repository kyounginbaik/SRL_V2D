"""R33: resting pose check (GT-free physics prior: at rest both objects lie on the tabletop). Per world (old 7b / latest r6), frames
f0-40 and f320-347: cup symmetry-axis tilt from the table normal, cup bottom height; spoon long-axis elevation, spoon flat-face normal
tilt from the table normal, bowl / handle-tip lowest heights. All in the scene's setup frame (tabletop z = 0)."""
import numpy as np, mujoco
from pathlib import Path
R = Path('/data1/hansol/retarget33')
def body_verts(m, d, n):
    b = m.body(n).id; out = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == b and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            a = m.mesh_vertadr[m.geom_dataid[g]]; v = m.mesh_vert[a:a + m.mesh_vertnum[m.geom_dataid[g]]].astype(float)
            out.append(v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
    return np.vstack(out)
for lab in ('R33_HGA7b', 'R33_HGA_r6'):
    W = R / 'worlds' / lab / 'ep33/episode_corrected/episode_000033'; m = mujoco.MjModel.from_xml_path(str(W / 'stage2/scene.xml')); d = mujoco.MjData(m)
    Q = np.load(W / 'reference_trajectory.npz', allow_pickle=True)['qpos']; print(f'\n=== {lab}')
    for f in (0, 20, 40, 320, 335, 347):
        d.qpos[:] = Q[f]; mujoco.mj_kinematics(m, d)
        X = body_verts(m, d, 'blue_cup'); c = X.mean(0); w, U = np.linalg.eigh(np.cov((X - c).T)); k = 0 if abs(np.log(w[1] / w[0])) > abs(np.log(w[2] / w[1])) else 2
        ax = U[:, k]; ax = ax if ax[2] > 0 else -ax; cup = f'cup: axis tilt {np.degrees(np.arccos(ax[2])):5.1f} deg, lowest {100*X[:,2].min():5.1f} cm, highest {100*X[:,2].max():5.1f}'
        S = body_verts(m, d, 'wooden_spoon'); c = S.mean(0); w, U = np.linalg.eigh(np.cov((S - c).T)); L = U[:, 2]; N = U[:, 0]; s = (S - c) @ L
        lo, hi = s.min(), s.max(); wid = lambda msk: np.linalg.norm((S[msk] - c) - np.outer(s[msk], L), axis=1).max()
        bowl = (s > hi - 0.3 * (hi - lo)) if wid(s > hi - 0.3 * (hi - lo)) > wid(s < lo + 0.3 * (hi - lo)) else (s < lo + 0.3 * (hi - lo))
        sp = (f'spoon: long-axis elevation {np.degrees(np.arcsin(abs(L[2]))):5.1f} deg, flat-face normal tilt {np.degrees(np.arccos(abs(N[2]))):5.1f} deg, '
              f'bowl lowest {100*S[bowl,2].min():5.1f} cm, handle lowest {100*S[~bowl,2].min():5.1f} cm, centre z {100*c[2]:5.1f}')
        print(f'  f{f:3d} {f/20:5.2f}s  {cup} | {sp}')
