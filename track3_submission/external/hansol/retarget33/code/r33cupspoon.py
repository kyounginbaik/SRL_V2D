"""R33: spoon-cup relation per frame, robot rollouts vs their targets (GT-free). The spoon bowl centre (from the spoon mesh: the wide end
of its long PCA axis, last 30 % of the length) is expressed in the cup frame (cup mesh: symmetry axis, radius, bottom and rim heights).
Reported per 0.5 s: bowl inside the cup (radial < R and between bottom and rim), depth below the rim, and the distance between the
robot's and the target's bowl centre in the cup frame (the RPE-like relation the tracking cost does not see).
usage: r33cupspoon.py"""
import json
import numpy as np
import mujoco
import trimesh
from pathlib import Path

K = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK33_ep33')
T = Path('/data1/hansol/retarget33')
c9 = json.load(open(K / 'runs/save_K33_cwHGAfix_ep33_HGA_c9/actor/evaluation/result.json'))['execution']
m = mujoco.MjModel.from_xml_path(c9['scene']); d = mujoco.MjData(m)


def geom_verts(body):
    b = m.body(body).id; out = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == b and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]; v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(float)
            out.append((g, v))
    return out


def spoon_bowl_local(v):
    c = v.mean(0); w, U = np.linalg.eigh(np.cov((v - c).T)); ax = U[:, 2]; s = (v - c) @ ax
    lo, hi = s.min(), s.max(); L = hi - lo
    def width(mask):
        p = (v[mask] - c) - np.outer(s[mask], ax); return np.linalg.norm(p, axis=1).max()
    a, b = s > hi - 0.3 * L, s < lo + 0.3 * L
    sel = a if width(a) > width(b) else b
    return v[sel].mean(0)


def cup_geom(v):
    c = v.mean(0); w, U = np.linalg.eigh(np.cov((v - c).T))
    k = 0 if abs(np.log(w[1] / w[0])) > abs(np.log(w[2] / w[1])) else 2   # isolated eigenvalue = symmetry axis
    ax = U[:, k]; h = (v - c) @ ax; r = np.linalg.norm((v - c) - np.outer(h, ax), axis=1); R = np.percentile(r, 95)
    lo, hi = h.min(), h.max(); near = r < 0.5 * R
    bottom_lo = near[h < lo + 0.15 * (hi - lo)].mean() > near[h > hi - 0.15 * (hi - lo)].mean()   # the closed end has vertices near the axis
    if not bottom_lo: ax, h, lo, hi = -ax, -h, -hi, -lo
    return dict(c=c, ax=ax, R=R, bottom=lo, rim=hi)


def relation(cw, cRot, cupg, bowl_w):
    rel = (bowl_w - cw)
    p = np.einsum('fji,fj->fi', cRot, rel) - cupg['c']                    # in cup local frame, relative to its centroid
    h = p @ cupg['ax']; r = np.linalg.norm(p - np.outer(h, cupg['ax']), axis=1)
    inside = (r < cupg['R']) & (h > cupg['bottom']) & (h < cupg['rim'])
    return p, r, h, inside


def robot(Q):
    (gs, vs), = geom_verts('wooden_spoon')[:1]; (gc, vc), = geom_verts('blue_cup')[:1]
    bl = spoon_bowl_local(vs); cg = cup_geom(vc)
    bw, cw, cR = [], [], []
    for q in Q:
        d.qpos[:] = q; mujoco.mj_kinematics(m, d)
        bw.append(d.geom_xpos[gs] + d.geom_xmat[gs].reshape(3, 3) @ bl); cw.append(d.geom_xpos[gc].copy()); cR.append(d.geom_xmat[gc].reshape(3, 3).copy())
    return relation(np.array(cw), np.array(cR), cg, np.array(bw)), cg


def target(tdir, n):
    e = tdir / 'episode_000033'
    vs = np.asarray(trimesh.load(e / 'wooden_spoon_mesh_metric.obj', force='mesh').vertices, float)
    vc = np.asarray(trimesh.load(e / 'blue_cup_mesh_metric.obj', force='mesh').vertices, float)
    Ts = np.load(e / 'wooden_spoon_trajectory.npz')['T_world_obj'][:n]; Tc = np.load(e / 'blue_cup_trajectory.npz')['T_world_obj'][:n]
    bl = spoon_bowl_local(vs); cg = cup_geom(vc)
    bw = Ts[:, :3, 3] + np.einsum('fij,j->fi', Ts[:, :3, :3], bl)
    return relation(Tc[:, :3, 3], Tc[:, :3, :3], cg, bw), cg


runs = [('HGA_c9 (team, official fail)', np.load(c9['trajectory'])['qpos'], K / 'targets/K33_cwHGAfix'),
        ('R2 (ours, official fail)', np.load(T / 'runs/eval_R2/eval.npz')['Q'], T / 'targets/R33_HGA7b'),
        ('R5 (ours, official fail)', np.load(T / 'runs/eval_R5/eval.npz')['Q'], T / 'targets/R33_HGA_r6')]
for lab, Q, tg in runs:
    (pr, rr, hr, ir), cgr = robot(Q); (pt, rt, ht, it), cgt = target(tg, len(Q))
    # compare in units of each cup's own geometry (depth below rim, radial) so mesh-frame offsets between scene and target meshes cancel
    dr = 100 * (cgr['rim'] - hr); dt = 100 * (cgt['rim'] - ht)
    gap = 100 * np.hypot(rr - rt, (hr - cgr['bottom']) - (ht - cgt['bottom']))
    print(f'\n== {lab}: cup R {100*cgr["R"]:.1f} cm, height {100*(cgr["rim"]-cgr["bottom"]):.1f} cm | target in-cup frames {int(it.sum())}, '
          f'robot in-cup {int(ir.sum())}, both {int((it & ir).sum())}, target-in but robot-out {int((it & ~ir).sum())} | bowl gap (cup frame) mean {gap.mean():.2f} cm, '
          f'on target-in frames {gap[it].mean() if it.any() else float("nan"):.2f} cm')
    print('   t     target: in depth_cm r_cm | robot: in depth_cm r_cm | gap_cm')
    for a in range(0, len(Q), 10):
        b = min(a + 10, len(Q)); s = slice(a, b)
        if it[s].any() or ir[s].any():
            print(f'   {a/20:5.1f}s   {int(it[s].sum()):2d}/{b-a} {dt[s].mean():6.1f} {100*rt[s].mean():5.1f} |   {int(ir[s].sum()):2d}/{b-a} {dr[s].mean():6.1f} {100*rr[s].mean():5.1f} | {gap[s].mean():5.2f}')
