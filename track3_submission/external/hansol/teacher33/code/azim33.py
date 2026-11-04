"""T33: which side of the cup does the LEFT hand hold? (GT-free). Azimuth of the left palm about the cup axis, seen from the camera:
phi = 0 cup front (towards the camera), -90 = cup's image-LEFT side, +90 = image-RIGHT side, 180 = behind. Also the right palm / spoon.
Human: W1 source hand_targets (MANO joints, palm = mean of wrist+MCPs) in the source frame of its object_poses (same frame).
Robot: rollout qpos (trajectory.npz key qpos, or eval.npz key Q) in the physics world, camera = setup_from_source @ world_from_rgb.
usage: azim33.py ROLLOUT.npz [...]"""
import sys, json, numpy as np, mujoco
from pathlib import Path
from scipy.spatial.transform import Rotation as R
W1 = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamB_ep3233/ep33/W1/episode_corrected/episode_000033')
man = json.loads((W1 / 'input_manifest.json').read_text()); src = Path(man['source']); SfS = np.asarray(man['setup_from_source'])
op = np.load(src / 'object_poses.npz'); print('object_poses keys', op.files)
Wfo = op['world_from_object']; names = [str(x) for x in op['names']]; ci = names.index('blue_cup'); si = names.index('wooden_spoon')
Wrgb = op['world_from_rgb'] if 'world_from_rgb' in op.files else None
CUPAX = np.array([-0.0237963577378219, -0.005046973670649213, 0.9997040869253169])   # teacher_ilc_b.py cup axis in the shipped body frame
def phi(o, ax, tocam):
    o = o - (o @ ax) * ax; f = tocam - (tocam @ ax) * ax; f /= np.linalg.norm(f); x = np.cross(ax, f)   # x = image-right if ax is up
    return np.degrees(np.arctan2(o @ x, o @ f))
def table(tag, t, phl, phr, extra=''):
    print(f'== {tag} {extra}'); print('  t(s)  left_phi  right_phi')
    for a in range(0, len(t), 20): print(f'  {t[a]:5.1f}  {phl[a]:7.1f}  {phr[a]:7.1f}')
# human (source frame); cup axis = up of the source cup at frame 0 is unknown in that body frame, so use the setup z (table normal) mapped back
ht = np.load(src / 'hand_targets.npz'); J = ht['hand_joints'].astype(float); n = len(J)
up_src = np.linalg.inv(SfS)[:3, :3] @ np.array([0, 0, 1.])
cam = Wrgb[:, :3, 3] if Wrgb is not None else np.zeros((n, 3))
cc = Wfo[:n, ci, :3, 3]
for h in range(J.shape[1]):
    palm = J[:, h, [0, 5, 9, 13, 17]].mean(1); ph = np.array([phi(palm[k] - cc[k], up_src, cam[k] - cc[k]) for k in range(n)])
    pr = np.array([phi(Wfo[k, si, :3, 3] - cc[k], up_src, cam[k] - cc[k]) for k in range(n)])
    table(f'HUMAN hand index {h} (valid {int(ht["hand_valid"][:, h].sum())})', np.arange(n) / 20, ph, pr, extra='(right column = spoon origin)')
    print('  median dist palm-cup cm', round(100 * np.median(np.linalg.norm(palm - cc, axis=1)), 1))
for f in sys.argv[1:]:
    z = np.load(f, allow_pickle=True); Q = z['qpos'] if 'qpos' in z.files else z['Q']
    sc = json.loads((Path(f).parent / 'execution_report.json').read_text()).get('scene') if (Path(f).parent / 'execution_report.json').exists() else None
    sc = sc or str(Path(f).parent / 'scene_physics.xml')
    m = mujoco.MjModel.from_xml_path(sc); d = mujoco.MjData(m); bl, br, bc, bs = (m.body(x).id for x in ('robot_left_left_hand_C_MC', 'robot_right_right_hand_C_MC', 'blue_cup', 'wooden_spoon'))
    camp = (SfS[:3, :3] @ cam.T).T + SfS[:3, 3]; N = len(Q); pl, pr = np.zeros(N), np.zeros(N); dl = np.zeros(N)
    for k in range(N):
        d.qpos[:] = Q[k]; mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d); c = d.xipos[bc]; ax = d.xmat[bc].reshape(3, 3) @ CUPAX
        ax = ax if ax[2] > 0 else -ax; kk = min(k, len(camp) - 1)
        pl[k] = phi(d.xpos[bl] - c, ax, camp[kk] - c); pr[k] = phi(d.xpos[br] - c, ax, camp[kk] - c); dl[k] = np.linalg.norm(d.xpos[bl] - c)
    table(f'ROBOT {f}', np.arange(N) / 20, pl, pr, extra=f'(left palm-cup dist median {100*np.median(dl):.1f} cm; right column = right palm)')
