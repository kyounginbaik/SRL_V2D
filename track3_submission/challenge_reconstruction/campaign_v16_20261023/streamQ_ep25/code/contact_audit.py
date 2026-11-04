"""Slot-aware penetration audit of a reference: place both boards at the reference poses in the physics scene (slot-cut collision,
+3 mm clearance, table) and report per frame the deepest board-board and board-table penetration (MuJoCo contact distance, mm).
usage: contact_audit.py POSES.npz OUT.json [SCENE]"""
import sys, json, numpy as np, mujoco
from scipy.spatial.transform import Rotation as R
scene = sys.argv[3] if len(sys.argv) > 3 else '/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamF_ep25/search/hLR_R_13.0_17.75_s3/execution/scene_physics.xml'
T = np.load(sys.argv[1])['world_from_object']; m = mujoco.MjModel.from_xml_path(scene); d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0)
names = ['wooden_piece_1', 'wooden_piece_2']; adr = {n: int(m.joint(n + '_free').qposadr[0]) for n in names}; bid = {n: m.body(n).id for n in names}
# park the robot far away so only object/object and object/table contacts count
rob = [j for j in range(m.njnt) if m.joint(j).name.startswith(('left', 'right'))]
for j in rob:
    if m.joint(j).name.endswith('pos_z'): d.qpos[m.jnt_qposadr[j]] += 5.
rows = []
for k in range(len(T)):
    for n in names:
        i = names.index(n); d.qpos[adr[n]:adr[n] + 3] = T[k, i, :3, 3]; d.qpos[adr[n] + 3:adr[n] + 7] = R.from_matrix(T[k, i, :3, :3]).as_quat(scalar_first=True)
    mujoco.mj_forward(m, d); bb = 0.; bt = {n: 0. for n in names}
    for c in d.contact[:d.ncon]:
        b1, b2 = m.body_rootid[m.geom_bodyid[c.geom1]], m.body_rootid[m.geom_bodyid[c.geom2]]
        s = {b1, b2}
        if s == {bid[names[0]], bid[names[1]]}: bb = min(bb, c.dist)
        for n in names:
            if bid[n] in s and (0 in s or m.body(int(b1 if b2 == bid[n] else b2)).name.lower().startswith(('table', 'world'))): bt[n] = min(bt[n], c.dist)
    rows.append(dict(t=k / 20, board_board_mm=round(-bb * 1000, 2), table_mm={n: round(-v * 1000, 2) for n, v in bt.items()}))
json.dump(rows, open(sys.argv[2], 'w'), indent=0)
for r in rows[::5]: print(f"{r['t']:5.2f} b1-b2 pen {r['board_board_mm']:6.1f} mm | table pen b1 {r['table_mm'][names[0]]:5.1f} b2 {r['table_mm'][names[1]]:5.1f}")
