"""settle test of the sweeper home keyframe (robot held at its keyframe via actuators = keyframe ctrl), original vs twin, with optional lift."""
import sys, json, numpy as np, mujoco
from scipy.spatial.transform import Rotation as R
from pathlib import Path
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamF_margins/ep19/code')
from common19F import EPDIR
GTW = np.array(json.loads(Path('/data1/SRL_V2D/challenge_reconstruction/recon_v15_20261022/episode_000019/change.json').read_text())['G'])
man = json.loads((EPDIR / 'input_manifest.json').read_text()); m = mujoco.MjModel.from_xml_path(man['physical_scene']); d = mujoco.MjData(m)
a = m.joint('mini_sweeper_free').qposadr[0]; b = m.body('mini_sweeper').id
geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_contype[g]]
def verts():
    out = []
    for g in geoms:
        mid = m.geom_dataid[g]; v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        out.append(v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
    return np.concatenate(out)
def run(twin, lift_mm, secs=2.0):
    mujoco.mj_resetDataKeyframe(m, d, 0)
    if twin:
        Tk = np.eye(4); Tk[:3, 3] = d.qpos[a:a + 3]; Tk[:3, :3] = R.from_quat(d.qpos[a + 3:a + 7], scalar_first=True).as_matrix(); T0 = Tk @ GTW
        d.qpos[a:a + 3] = T0[:3, 3]; d.qpos[a + 3:a + 7] = R.from_quat(R.from_matrix(T0[:3, :3]).as_quat(), scalar_first=False).as_quat(scalar_first=True)
    mujoco.mj_forward(m, d); z0 = verts()[:, 2].min()
    tz = None
    # table top height: lowest point of the original keyframe sweeper ~ table
    d.qpos[a + 2] += lift_mm / 1000; mujoco.mj_forward(m, d); p0 = d.qpos[a:a + 3].copy(); q0 = d.qpos[a + 3:a + 7].copy()
    if m.nkey and m.key_ctrl.size: d.ctrl[:] = m.key_ctrl[0]
    n = int(secs / m.opt.timestep)
    for _ in range(n): mujoco.mj_step(m, d)
    dp = np.linalg.norm(d.qpos[a:a + 3] - p0) * 100; dr = np.degrees((R.from_quat(d.qpos[a + 3:a + 7], scalar_first=True) * R.from_quat(q0, scalar_first=True).inv()).magnitude())
    return z0, dp, dr, d.qpos[a:a + 7].copy()
for twin in (0, 1):
    for lift in ([0] if not twin else [0, 5, 8, 11, 14]):
        z0, dp, dr, q = run(twin, lift); print('twin', twin, 'lift', lift, 'lowest z %.2f cm' % (z0 * 100), 'settle drift %.2f cm %.1f deg' % (dp, dr), flush=True)
