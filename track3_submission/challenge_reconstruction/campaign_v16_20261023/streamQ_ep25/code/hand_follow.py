"""Stream Q phase 6: move a teacher's wrist commands with the object it handles when the reference objects move (v21 reference ->
compose reference). Per frame d_i(t) = slab-centre displacement of board i (new - old reference); left wrist += smooth(d_1) from
L0 s (blend 0.5 s), right wrist += smooth(d_2) from R0 s; optional rotation of each wrist by the board's in-plane yaw change
(about the board centre). Fingers unchanged. Executes in the world scene, scores vs the world reference.
usage: hand_follow.py SRC_DIR OLD_POSES.npz WORLD_DIR OUT_DIR L0 R0 [YAW=0|1] [SCENE]"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import sys, json
from pathlib import Path
import numpy as np, mujoco
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R
LEG = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017')
sys.path[:0] = [str(LEG / 'code'), '/data1/SRL_V2D/third_party/DexAgent', str(Path(__file__).resolve().parent)]
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files
import correct_ep25_v2 as C
src, oldp, world, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]); L0, R0 = float(sys.argv[5]), float(sys.argv[6])
YAW = int(sys.argv[7]) if len(sys.argv) > 7 else 0; scene = Path(sys.argv[8]) if len(sys.argv) > 8 else world / 'scene/scene_physics_C.xml'
epdir = world / 'episodes/episode_000025'; man = json.loads((epdir / 'input_manifest.json').read_text()); out.mkdir(parents=True, exist_ok=False)
F = C.frames(Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/collision_recon_slots_v3/episode_000025'))
To = np.load(oldp)['world_from_object']; Tn = np.load(world / 'source_corrected/episode_000025/object_poses.npz')['world_from_object']
z = {k: v for k, v in np.load(src / 'reference.npz', allow_pickle=False).items()}; Q = np.asarray(z['qpos'], float).copy(); names = list(np.asarray(z['joint_names']).astype(str))
m = mujoco.MjModel.from_xml_path(str(scene))
from tool_library.common import tracking as TR
tgt = TR.load_reference(epdir / 'tracking_reference.npz', m)['qpos']
for o in ('wooden_piece_1', 'wooden_piece_2'):
    adr = int(m.joint(o + '_free').qposadr[0]); Q[:, adr:adr + 7] = tgt[:, adr:adr + 7]
sm = lambda u: np.clip(u, 0, 1) ** 2 * (3 - 2 * np.clip(u, 0, 1)); t = np.arange(len(Q)) / 20.
for i, (side, t0) in enumerate((('left', L0), ('right', R0))):
    f = F[('wooden_piece_1', 'wooden_piece_2')[i]]; c = f['slab_c']
    co = np.einsum('kij,j->ki', To[:, i, :3, :3], c) + To[:, i, :3, 3]; cn = np.einsum('kij,j->ki', Tn[:, i, :3, :3], c) + Tn[:, i, :3, 3]
    d = gaussian_filter1d(cn - co, 3.0, axis=0, mode='nearest'); w = sm((t - (t0 - .5)) / .5)
    cols = [names.index(f'robot_{side}_{side}_{a}') for a in ('pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z')]
    if YAW:   # yaw change of the board's long axis (sign-free), about the board centre
        lo = To[:, i, :3, :3] @ f['U'][:, 2]; ln = Tn[:, i, :3, :3] @ f['U'][:, 2]
        yaw = np.arctan2(lo[:, 0] * ln[:, 1] - lo[:, 1] * ln[:, 0], (lo[:, :2] * ln[:, :2]).sum(1)); yaw = np.where(yaw > np.pi / 2, yaw - np.pi, np.where(yaw < -np.pi / 2, yaw + np.pi, yaw)); yaw = gaussian_filter1d(yaw, 3.0)
        for k in range(len(Q)):
            Rk = R.from_rotvec([0, 0, w[k] * yaw[k]]); p = Q[k, cols[:3]]; Q[k, cols[:3]] = co[k] + Rk.apply(p - co[k]); Q[k, cols[3:6]] = (Rk * R.from_euler('XYZ', Q[k, cols[3:6]])).as_euler('XYZ')
        for cc in cols[3:6]: Q[:, cc] = np.unwrap(Q[:, cc])
    Q[:, cols[:3]] += w[:, None] * d
z['qpos'] = Q; np.savez(out / 'reference.npz', **z)
ex = execute_reference(str(scene), str(out / 'reference.npz'), out / 'execution')
met = evaluate_files(man['source'], ex['trajectory'], epdir / 'input_manifest.json', ex['scene'], epdir / 'metric_geometry.json')
fmt = {k: (round(v['mean_position_error_cm'][0], 3), round(v['mean_orientation_error_degrees'][0], 2), round(v['max_position_error_m'][0] * 100, 1)) for k, v in met['official_style_reference_proxy']['objects'].items()}
res = dict(episode='episode_000025', input_dir=str(epdir), reference=str(out / 'reference.npz'), execution=ex, metrics=met, parent=str(src), hand_follow=dict(old=str(oldp), L0=L0, R0=R0, yaw=YAW), learned_policy=False)
(out / 'result.json').write_text(json.dumps(res, indent=1, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)) + '\n')
print(json.dumps(dict(out=str(out), mp_sr=met['reference_proxy_mp_sr'], objects=fmt, verified=ex['physics_verified'])), flush=True)
