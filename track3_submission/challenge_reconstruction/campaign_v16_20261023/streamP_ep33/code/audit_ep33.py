"""Stream P, episode 33: per-frame physical-contradiction audit of a reference object track (default: the ORIGINAL
final_recon object_poses.npz), in the physics setup frame (table top z = 0).
 - spoon vs cup: spoon visual surface samples expressed in the cup body frame (cup axis = body z, cavity opening at -z,
   visual inner floor z = +0.0775 m, visual outer bottom z = +0.0880 m, collision floor top z = +0.067 m).
   floor_pen_mm = how far spoon points that are inside the cavity radius go beyond the inner floor (>0 = into/through the floor);
   below_bottom_mm = beyond the outer bottom (through the cup). wall_pen_mm = depth of spoon points inside the wall annulus.
   In addition the MuJoCo collision depth between the spoon and cup collision parts (same parts as the physics scene).
 - spoon/cup vs table: lowest visual point z (negative = below the table top).
 - spoon vs hand: MANO stereo hand vertices inside the spoon convex collision parts (max depth mm, vertex count).
usage: audit_ep33.py OUT_PREFIX [object_poses.npz]"""
import sys, json, numpy as np, trimesh, mujoco
from scipy.spatial.transform import Rotation as Rot
FR = '/data1/SRL_V2D/challenge_reconstruction/final_recon/episode_000033'
EPD = '/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/episodes/episode_000033'
SCENE = '/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamJ_ep3233/ep33/rewrap/start16.2_open1.0/evaluation/execution/scene_physics.xml'
out = sys.argv[1]; poses = sys.argv[2] if len(sys.argv) > 2 else FR + '/object_poses.npz'
man = json.load(open(EPD + '/input_manifest.json')); S = np.asarray(man['setup_from_source'])
z = np.load(poses); names = [str(x) for x in z['names']]; W = z['world_from_object']; n = len(W)
P = S @ W; Pc, Ps = P[:, names.index('blue_cup')], P[:, names.index('wooden_spoon')]
cup = trimesh.load(FR + '/assets/blue_cup_visual.obj', force='mesh'); sp = trimesh.load(FR + '/assets/wooden_spoon_visual.obj', force='mesh')
sv = sp.sample(6000, seed=0) if 'seed' in trimesh.sample.sample_surface.__code__.co_varnames else sp.sample(6000)
cv = cup.vertices
# cavity radius profile (inner wall ~ 10th percentile radius per slice) in the cup frame
zs = np.linspace(-0.083, 0.075, 40); rc = np.hypot(cv[:, 0], cv[:, 1])
rin = np.array([np.percentile(rc[(abs(cv[:, 2] - q) < 0.006) & (rc > 0.03)], 10) for q in zs])
rout = np.array([np.percentile(rc[(abs(cv[:, 2] - q) < 0.006) & (rc > 0.03)], 95) for q in zs])
FLOOR_IN, BOTTOM, COLL_FLOOR, RIM = 0.0775, 0.088, 0.067, -0.083
# hands (camera frame -> world via world_from_rgb)
import glob
from scipy.spatial import ConvexHull
HULLS = [ConvexHull(trimesh.load(f, force='mesh').vertices).equations for f in sorted(glob.glob(FR + '/assets/wooden_spoon_convex/*.obj'))]
H = np.load(FR + '/hands_stereo.npz'); Wc = z['world_from_rgb']
# MuJoCo collision depth spoon<->cup and object<->table at the reference poses
m = mujoco.MjModel.from_xml_path(SCENE); d = mujoco.MjData(m); d.qpos[:] = m.key_qpos[0] if m.nkey else m.qpos0
adr = {k: int(m.jnt_qposadr[m.body(k).jntadr[0]]) for k in names}; bid = {m.body(k).id: k for k in names}; tab = m.geom('g_table').id
rows = []
for k in range(n):
    rel = np.linalg.inv(Pc[k]) @ Ps[k]; q = sv @ rel[:3, :3].T + rel[:3, 3]; r = np.hypot(q[:, 0], q[:, 1])
    ri = np.interp(np.clip(q[:, 2], zs[0], zs[-1]), zs, rin); ro = np.interp(np.clip(q[:, 2], zs[0], zs[-1]), zs, rout)
    incav = (r < ri - 0.002) & (q[:, 2] > RIM)
    floor = float(np.max(q[incav, 2] - FLOOR_IN, initial=-1)) * 1000 if incav.any() else np.nan
    collfloor = float(np.max(q[incav, 2] - COLL_FLOOR, initial=-1)) * 1000 if incav.any() else np.nan
    underfoot = (r < 0.035); below = float(np.max(q[underfoot, 2] - BOTTOM, initial=-1)) * 1000 if underfoot.any() else np.nan
    inwall = (r >= ri) & (r <= ro) & (q[:, 2] > RIM) & (q[:, 2] < FLOOR_IN)
    wall = float(np.max(np.minimum(r[inwall] - ri[inwall], ro[inwall] - r[inwall]), initial=0)) * 1000
    bowl = q[sv[:, 0] > 0.06]; bowl_c = bowl.mean(0)
    spw = sv @ Ps[k][:3, :3].T + Ps[k][:3, 3]; cpw = cv @ Pc[k][:3, :3].T + Pc[k][:3, 3]
    # hands
    hpen, hcount = 0.0, 0
    for side in range(2):
        if not H['valid'][k, side]: continue
        hv = H['vertices_camera_m'][k, side] @ Wc[k][:3, :3].T + Wc[k][:3, 3]; hv = hv @ S[:3, :3].T + S[:3, 3]
        hl = (hv - Ps[k][:3, 3]) @ Ps[k][:3, :3]
        near = np.abs(hl).max(1) < 0.17
        if near.any():
            for E in HULLS:
                dist = hl[near] @ E[:, :3].T + E[:, 3]; inside = (dist < 0).all(1)
                if inside.any(): hpen = max(hpen, float((-dist[inside].max(1)).max()) * 1000); hcount += int(inside.sum())
    for nm, T in zip(names, P[k]):
        d.qpos[adr[nm]:adr[nm] + 3] = T[:3, 3]; d.qpos[adr[nm] + 3:adr[nm] + 7] = Rot.from_matrix(T[:3, :3]).as_quat()[[3, 0, 1, 2]]
    mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d); mujoco.mj_collision(m, d)
    oo = 0.0; ot = {nm: 0.0 for nm in names}
    for c in d.contact[:d.ncon]:
        b1, b2 = bid.get(int(m.geom_bodyid[c.geom1])), bid.get(int(m.geom_bodyid[c.geom2])); dep = max(0., -float(c.dist)) * 1000
        if b1 and b2 and b1 != b2: oo = max(oo, dep)
        elif b1 and c.geom2 == tab: ot[b1] = max(ot[b1], dep)
        elif b2 and c.geom1 == tab: ot[b2] = max(ot[b2], dep)
    cup_up = -Pc[k][:3, 2]  # opening direction (body -z) in setup frame
    rows.append(dict(t=k / 20.0, floor_pen_mm=floor, collfloor_pen_mm=collfloor, below_bottom_mm=below, wall_pen_mm=wall,
        bowl_depth_below_rim_cm=float((bowl_c[2] - RIM) * 100), bowl_r_cm=float(np.hypot(*bowl_c[:2]) * 100), spoon_in_cavity_frac=float(incav.mean()),
        mj_spoon_cup_mm=oo, mj_cup_table_mm=ot['blue_cup'], mj_spoon_table_mm=ot['wooden_spoon'],
        spoon_min_z_cm=float(spw[:, 2].min() * 100), cup_min_z_cm=float(cpw[:, 2].min() * 100), cup_tilt_deg=float(np.degrees(np.arccos(np.clip(cup_up[2], -1, 1)))),
        cup_xyz_cm=(Pc[k][:3, 3] * 100).round(2).tolist(), spoon_xyz_cm=(Ps[k][:3, 3] * 100).round(2).tolist(), hand_in_spoon_mm=hpen, hand_in_spoon_n=hcount))
json.dump(rows, open(out + '.json', 'w'), indent=0)
with open(out + '.txt', 'w') as f:
    f.write('  t   floor coll  below  wall  bowlDep bowlR inCav | mj_sc mj_ct mj_st | spMinZ cupMinZ tilt | handPen n\n')
    for r in rows:
        f.write(f"{r['t']:5.2f} {r['floor_pen_mm']:6.1f} {r['collfloor_pen_mm']:5.1f} {r['below_bottom_mm']:6.1f} {r['wall_pen_mm']:5.1f}  {r['bowl_depth_below_rim_cm']:6.2f} {r['bowl_r_cm']:5.2f} {r['spoon_in_cavity_frac']:4.2f} | "
                f"{r['mj_spoon_cup_mm']:5.1f} {r['mj_cup_table_mm']:5.1f} {r['mj_spoon_table_mm']:5.1f} | {r['spoon_min_z_cm']:6.2f} {r['cup_min_z_cm']:6.2f} {r['cup_tilt_deg']:5.1f} | {r['hand_in_spoon_mm']:5.1f} {r['hand_in_spoon_n']}\n")
print('wrote', out)
