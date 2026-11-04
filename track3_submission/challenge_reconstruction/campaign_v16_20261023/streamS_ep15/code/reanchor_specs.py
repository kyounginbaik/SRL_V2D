"""Stream S: re-express pickup-search specs (already ported to a world by port_specs.py) on a world whose spoon reference moved
(world_6prong_nt, NEW spoon track): the right-hand anchor wrist pose is moved rigidly with the spoon's frame-0 reference pose
(D = W_new[0] @ inv(W_old[0])), so the grasp stays the same relative to the spoon; the anchor's object qpos becomes the new
frame-0 reference. Search parameters are offsets about the anchor, so a resumed progress.json keeps the same relative grasp.
Optional --release T / --times JSON as port_specs.py. usage: reanchor_specs.py OLD_WORLD NEW_WORLD OUT_DIR SPEC.json [...] [--suffix S]"""
import json, sys, argparse
from pathlib import Path
import numpy as np, mujoco
from scipy.spatial.transform import Rotation as R
LEG = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017'); sys.path[:0] = ['/data1/SRL_V2D/third_party/DexAgent', str(LEG / 'code')]
from tool_library.common import hand as H, tracking as TR
from wrist_reference import continuous_euler_xyz
ap = argparse.ArgumentParser(); ap.add_argument('old'); ap.add_argument('new'); ap.add_argument('out'); ap.add_argument('specs', nargs='+')
ap.add_argument('--release', type=float); ap.add_argument('--times'); ap.add_argument('--suffix', default=''); a = ap.parse_args()
Wo, Wn = Path(a.old).resolve(), Path(a.new).resolve(); EPn = Wn / 'episodes/episode_000015'; out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
scene = str(Wn / 'scene/controller_scene.xml'); m = mujoco.MjModel.from_xml_path(scene); man = json.loads((EPn / 'input_manifest.json').read_text())
qo = TR.load_reference(Wo / 'episodes/episode_000015/tracking_reference.npz', m)['qpos']; qn = TR.load_reference(EPn / 'tracking_reference.npz', m)['qpos']
oc = int(m.joint('wooden_spoon_free').qposadr[0])
def T(q): M = np.eye(4); M[:3, :3] = R.from_quat(q[oc + 3:oc + 7], scalar_first=True).as_matrix(); M[:3, 3] = q[oc:oc + 3]; return M
D = T(qn[0]) @ np.linalg.inv(T(qo[0]))
rig = H.HandRig.build(m, 'robot_right_', 'right'); wc = np.array([rig.wrist_qpos[k] for k in ('x', 'y', 'z', 'rx', 'ry', 'rz')]); made = []
for sp in a.specs:
    s = json.loads(Path(sp).read_text()); A = np.load(s['anchor_pose'])['qpos'].copy(); B = qn[0].copy()
    B[wc[:3]] = D[:3, :3] @ A[wc[:3]] + D[:3, 3]
    B[wc[3:]] = continuous_euler_xyz(R.from_matrix((R.from_matrix(D[:3, :3]) * R.from_euler('XYZ', A[wc[3:]])).as_matrix()[None]), A[wc[3:]])[0]
    keep = np.ones(len(A), bool); keep[oc:oc + 7] = False; rc = int(m.joint('plastic_dish_rack_free').qposadr[0]); keep[rc:rc + 7] = False; keep[wc] = False
    B[keep] = A[keep]                                      # fingers / left hand as in the anchor
    name = Path(sp).stem + a.suffix; ap_ = out / f'anchor_{name}.npz'; np.savez_compressed(ap_, qpos=B)
    s.update(scene=scene, original_reference=str(EPn / 'tracking_reference.npz'), geometry=str(EPn / 'metric_geometry.json'), manifest=str(EPn / 'input_manifest.json'),
             source=man['source'], anchor_pose=str(ap_), reanchored_from=str(Path(sp).resolve()), reanchor_D=D.tolist())
    if a.times: s['times'] = dict(s['times'], **json.loads(a.times))
    if a.release is not None: s['times'] = dict(s['times'], release=a.release); s['right_hand_holds_until_end'] = a.release >= s['horizon_s']
    p = out / f'{name}.json'
    if p.exists(): raise SystemExit(f'exists: {p}')
    p.write_text(json.dumps(s, indent=1) + '\n'); made.append(str(p))
print('D translation cm', np.round(D[:3, 3] * 100, 2), 'rot deg', round(float(np.degrees(R.from_matrix(D[:3, :3]).magnitude())), 2)); print(made)
