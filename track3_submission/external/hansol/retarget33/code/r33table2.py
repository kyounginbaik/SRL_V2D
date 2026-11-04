"""R33: objects vs the tabletop in each target world's own scene (setup/scene.xml, reference qpos = what the render's target panel shows):
lowest vertex height above the tabletop (z = 0), cup axis tilt from the table normal, cup centre distance to the table centre / edge."""
import numpy as np, mujoco
from pathlib import Path
R = Path('/data1/hansol/retarget33')
def cupax(v):
    c = v.mean(0); w, U = np.linalg.eigh(np.cov((v - c).T)); k = 0 if abs(np.log(w[1] / w[0])) > abs(np.log(w[2] / w[1])) else 2
    ax = U[:, k]; h = (v - c) @ ax; r = np.linalg.norm((v - c) - np.outer(h, ax), axis=1); R_ = np.percentile(r, 95); lo, hi = h.min(), h.max(); near = r < 0.5 * R_
    return ax if near[h < lo + 0.15 * (hi - lo)].mean() > near[h > hi - 0.15 * (hi - lo)].mean() else -ax
import sys
for lab in (sys.argv[1:] or ['R33_HGA7b', 'R33_HGA_r6']):
    W = R / 'worlds' / lab / 'ep33/episode_corrected/episode_000033'; m = mujoco.MjModel.from_xml_path(str(W / 'stage2/scene.xml')); d = mujoco.MjData(m)
    z = np.load(W / 'reference_trajectory.npz', allow_pickle=True); Q = z['qpos']; tg = m.geom('g_table').id
    print(f'\n=== {lab}: table geom pos {np.round(100*m.geom_pos[tg],1).tolist()} cm size {np.round(100*m.geom_size[tg],1).tolist()} cm')
    G = {}
    for n in ('blue_cup', 'wooden_spoon'):
        b = m.body(n).id; G[n] = [(g, m.mesh_vert[m.mesh_vertadr[m.geom_dataid[g]]: m.mesh_vertadr[m.geom_dataid[g]] + m.mesh_vertnum[m.geom_dataid[g]]].astype(float))
                                  for g in range(m.ngeom) if m.geom_bodyid[g] == b and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
    cv = np.vstack([v for _, v in G['blue_cup']]) if len(G['blue_cup']) == 1 else None
    ax = cupax(G['blue_cup'][0][1]) if cv is not None else None
    for f in (0, 10, 20, 30, 40, 60, 80, 120, 200, 280, 330, 347):
        d.qpos[:] = Q[f]; mujoco.mj_kinematics(m, d); s = f'  f{f:3d} {f/20:5.2f}s'
        for n in ('blue_cup', 'wooden_spoon'):
            X = np.vstack([v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g] for g, v in G[n]]); s += f' | {n[:5]} centre {np.round(100*d.xpos[m.body(n).id],1).tolist()} low z {100*X[:,2].min():5.1f}'
            if n == 'blue_cup' and ax is not None:
                g0 = G[n][0][0]; a = d.geom_xmat[g0].reshape(3, 3) @ ax; s += f' tilt {np.degrees(np.arccos(np.clip(a[2], -1, 1))):5.1f}'
        print(s)
