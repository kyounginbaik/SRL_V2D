"""Stream P ep33: minimal rigid translation that removes the spoon<->cup (or object<->table) collision-part penetration of a
reference frame, using the physics scene's own collision geometry (MuJoCo contacts, iterative push along the deepest contact
normal). Poses are setup-frame 4x4."""
import numpy as np, mujoco
from scipy.spatial.transform import Rotation as Rot
SCENE = '/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamJ_ep3233/ep33/rewrap/start16.2_open1.0/evaluation/execution/scene_physics.xml'
class Collider:
    def __init__(self, scene=SCENE):
        self.m = mujoco.MjModel.from_xml_path(scene); self.d = mujoco.MjData(self.m)
        self.d.qpos[:] = self.m.key_qpos[0] if self.m.nkey else self.m.qpos0
        self.names = ['blue_cup', 'wooden_spoon']; self.adr = {k: int(self.m.jnt_qposadr[self.m.body(k).jntadr[0]]) for k in self.names}
        self.bid = {self.m.body(k).id: k for k in self.names}; self.tab = self.m.geom('g_table').id
        # park the robot far away so only object/table contacts matter
        for j in range(self.m.njnt):
            if self.m.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE and (self.m.joint(j).name or '').startswith('robot'):
                self.d.qpos[self.m.jnt_qposadr[j]] += 5.0
    def contacts(self, poses):
        for nm, T in poses.items():
            a = self.adr[nm]; self.d.qpos[a:a + 3] = T[:3, 3]; self.d.qpos[a + 3:a + 7] = Rot.from_matrix(T[:3, :3]).as_quat()[[3, 0, 1, 2]]
        mujoco.mj_kinematics(self.m, self.d); mujoco.mj_comPos(self.m, self.d); mujoco.mj_collision(self.m, self.d); out = []
        for c in self.d.contact[:self.d.ncon]:
            b1, b2 = self.bid.get(int(self.m.geom_bodyid[c.geom1])), self.bid.get(int(self.m.geom_bodyid[c.geom2]))
            o1 = b1 or ('table' if c.geom1 == self.tab else None); o2 = b2 or ('table' if c.geom2 == self.tab else None)
            if o1 and o2 and o1 != o2 and c.dist < 0: out.append((o1, o2, -float(c.dist), c.frame[:3].copy()))
        return out
    def depth(self, poses, a, b):
        return max([dp for o1, o2, dp, n in self.contacts(poses) if {o1, o2} == {a, b}], default=0.0)
    def push_out(self, poses, mover, other, tol=0.0005, iters=60):
        """translate poses[mover] until its penetration with `other` is <= tol; returns new 4x4 and the translation."""
        P = {k: v.copy() for k, v in poses.items()}; t = np.zeros(3)
        for _ in range(iters):
            cs = [(dp, n if o1 == other else -n) for o1, o2, dp, n in self.contacts(P) if {o1, o2} == {mover, other}]
            if not cs or max(c[0] for c in cs) <= tol: break
            dp, n = max(cs, key=lambda c: c[0])           # normal points from geom1 to geom2; oriented from other -> mover
            step = n * (dp + 0.0003); P[mover][:3, 3] += step; t += step
        return P[mover], t
