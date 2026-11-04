"""Search a physically stable STANDING rest pose of the half-turn twin of the sweeper (yaw 180 about its long axis, which stands ~vertical):
twin = home @ G; candidate = Rtilt(about the base contact centre, world x/y tilt) @ twin, placed with lowest collision point at z=+clear; settle 2 s
with the robot held at its keyframe; record drift. Prints the best candidates (drift small, close to the twin)."""
import sys, json, itertools, numpy as np, mujoco
from scipy.spatial.transform import Rotation as R
from pathlib import Path
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamF_margins/ep19/code')
from common19F import EPDIR
GTW = np.array(json.loads(Path('/data1/SRL_V2D/challenge_reconstruction/recon_v15_20261022/episode_000019/change.json').read_text())['G'])
man = json.loads((EPDIR / 'input_manifest.json').read_text()); m = mujoco.MjModel.from_xml_path(man['physical_scene']); d = mujoco.MjData(m)
a = m.joint('mini_sweeper_free').qposadr[0]; b = m.body('mini_sweeper').id
geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_contype[g]]
Pb = []
for g in geoms:
    mid = m.geom_dataid[g]; v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
    gm = np.zeros(9); mujoco.mju_quat2Mat(gm, m.geom_quat[g]); Pb.append(v @ gm.reshape(3, 3).T + m.geom_pos[g])
Pb = np.concatenate(Pb)
mujoco.mj_resetDataKeyframe(m, d, 0); Tk = np.eye(4); Tk[:3, 3] = d.qpos[a:a + 3]; Tk[:3, :3] = R.from_quat(d.qpos[a + 3:a + 7], scalar_first=True).as_matrix()
Tw = Tk @ GTW; clear = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0005
def place(rx, ry):
    W = Pb @ Tw[:3, :3].T + Tw[:3, 3]; c = W[W[:, 2] < W[:, 2].min() + 0.01].mean(0)
    Rt = R.from_euler('xy', [rx, ry], degrees=True).as_matrix(); T = np.eye(4); T[:3, :3] = Rt @ Tw[:3, :3]; T[:3, 3] = Rt @ (Tw[:3, 3] - c) + c
    W2 = Pb @ T[:3, :3].T + T[:3, 3]; T[2, 3] += clear - W2[:, 2].min(); return T
def settle(T, secs=2.0):
    mujoco.mj_resetDataKeyframe(m, d, 0); d.qpos[a:a + 3] = T[:3, 3]; d.qpos[a + 3:a + 7] = R.from_matrix(T[:3, :3]).as_quat(scalar_first=True); d.qvel[:] = 0
    if m.key_ctrl.size: d.ctrl[:] = m.key_ctrl[0]
    mujoco.mj_forward(m, d)
    for _ in range(int(secs / m.opt.timestep)): mujoco.mj_step(m, d)
    Ts = np.eye(4); Ts[:3, 3] = d.qpos[a:a + 3]; Ts[:3, :3] = R.from_quat(d.qpos[a + 3:a + 7], scalar_first=True).as_matrix()
    return Ts, np.linalg.norm(Ts[:3, 3] - T[:3, 3]) * 100, np.degrees(R.from_matrix(Ts[:3, :3] @ T[:3, :3].T).magnitude())
res = []
rng = np.arange(-12, 12.1, 2.0)
for rx, ry in itertools.product(rng, rng):
    T = place(rx, ry); Ts, dp, dr = settle(T); off = np.degrees(R.from_matrix(Ts[:3, :3] @ Tw[:3, :3].T).magnitude())
    res.append((rx, ry, dp, dr, off, np.linalg.norm(Ts[:3, 3] - Tw[:3, 3]) * 100)); print('%5.1f %5.1f drift %.2f cm %.1f deg | settled vs twin %.1f deg %.2f cm' % res[-1], flush=True)
ok = sorted([r for r in res if r[3] < 3], key=lambda r: r[4]); print('BEST', ok[:5])
np.save(Path(__file__).resolve().parent.parent / 'twin_rest_grid.npy', np.array(res))
