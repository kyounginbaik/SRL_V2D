"""R33: end-of-clip motion diagnostic for one saved actor rollout (GT-free): per 0.1 s from 14.5 s, right-wrist joint speed (pos cm/s,
rot deg/s), right-finger joint speed (deg/s), spoon linear / angular speed, spoon height above the table, number of right-hand geoms
touching the spoon, spoon-table contact. usage: r33endjit.py RESULT_JSON"""
import sys, json, numpy as np, mujoco
from scipy.spatial.transform import Rotation as Rt
ex = json.load(open(sys.argv[1]))['execution']; m = mujoco.MjModel.from_xml_path(ex['scene']); d = mujoco.MjData(m)
z = np.load(ex['trajectory'], allow_pickle=True); Q = z['qpos'] if 'qpos' in z.files else z['Q']; n = len(Q); fps = 20.0
bs = m.body('wooden_spoon').id; tb = m.body('table').id if 'table' in [m.body(i).name for i in range(m.nbody)] else -1
def root(b):
    while m.body_parentid[b] != 0: b = m.body_parentid[b]
    return b
rr = root(m.body('robot_right_right_hand_C_MC').id)
jn = [m.joint(j).name for j in range(m.njnt)]; adr = m.jnt_qposadr
wr = [adr[j] for j, s in enumerate(jn) if s.startswith('robot_right_right_') and any(k in s for k in ('pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z')) and m.jnt_type[j] in (2, 3)][:6]; fg = [adr[j] for j, s in enumerate(jn) if s.startswith('robot_right_right_') and adr[j] not in wr and m.jnt_type[j] in (2, 3)]
sp = adr[[j for j, s in enumerate(jn) if m.jnt_bodyid[j] == bs][0]]
ps = Q[:, sp:sp + 3]; Rs = Rt.from_quat(Q[:, sp + 3:sp + 7], scalar_first=True)
vs = 100 * np.linalg.norm(np.diff(ps, axis=0), axis=1) * fps; ws = np.degrees((Rs[1:] * Rs[:-1].inv()).magnitude()) * fps
W = Q[:, wr]; vw = np.abs(np.diff(W, axis=0)) * fps; F = Q[:, fg]; vf = np.degrees(np.abs(np.diff(F, axis=0))).mean(1) * fps
nc = np.zeros(n, int); ct = np.zeros(n, int)
for k in range(n):
    d.qpos[:] = Q[k]; mujoco.mj_forward(m, d); s = set()
    for i in range(d.ncon):
        c = d.contact[i]; b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        for a, b, g in ((b1, b2, c.geom2), (b2, b1, c.geom1)):
            if a == bs and root(b) == rr: s.add(g)
            if a == bs and b == tb: ct[k] = 1
    nc[k] = len(s)
print('wrist qpos adr', wr, 'n finger joints', len(fg), [s for s in jn if s.startswith('robot_right_right_')][:8])
print('  t     | wrist pos cm/s  wrist rot deg/s | fingers deg/s | spoon cm/s  deg/s | spoon z cm | R geoms on spoon | spoon on table')
for a in range(236, 292, 2):
    s = slice(a, min(a + 2, n - 1))
    print(f' {a/20:5.2f} | {100*vw[s,:3].max():8.1f}  {np.degrees(vw[s,3:6]).max():10.1f} | {vf[s].max():8.1f} | {vs[s].max():7.1f} {ws[s].max():7.1f} | {100*ps[a,2]:6.1f} | {nc[a]:2d} | {ct[a]}')
