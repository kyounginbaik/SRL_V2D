"""Stream R ep32: pickup/grasp specs (physics_pickup_search_v2) whose post-closure wrist follows the CORRECTED spoon roll (V2a corrected episode dir:
its tracking_reference.npz spoon columns are the corrected targets). Seed reference = the v19 teacher (Stream J chainB/c2) commands with the
spoon target columns swapped for the corrected ones (object columns are targets only). Anchors = the teacher's achieved right-hand hold at
T_HELD re-expressed on the resting spoon (frame-0 pose), plus variants: power wrap (middle/ring/pinky flexion to FRAC of range), shifts along the
spoon long axis (body x, +x toward the bowl), rotations about it. Times: approach 1.3 -> hover 1.8 -> contact 2.1 -> closed 2.35 -> transport 2.55,
release RELEASE s; full horizon.
usage: prepare_pickup32.py OUT_DIR CORRECTED_EPISODE_DIR [--held 3.0] [--release 11.0] [--horizon 12.9 (search scoring horizon)]"""
import json, sys, argparse
from pathlib import Path
import numpy as np, mujoco
from scipy.spatial.transform import Rotation as R
LEG = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017'); sys.path[:0] = ['/data1/SRL_V2D/third_party/DexAgent', str(LEG / 'code')]
from tool_library.common import hand as H, tracking as TR
from wrist_reference import continuous_euler_xyz
from physics_pickup_search_v2 import dump
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('ced'); ap.add_argument('--held', type=float, default=3.0); ap.add_argument('--release', type=float, default=11.0); ap.add_argument('--horizon', type=float, default=12.9)
ap.add_argument('--teacher', default='/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamJ_ep3233/ep32/chainB/c2_R_9.0_11.0'); a = ap.parse_args()
out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True); ced = Path(a.ced).resolve(); TE = Path(a.teacher)
r = json.loads((TE / 'result.json').read_text()); scene = r['execution']['scene']; man = json.loads((ced / 'input_manifest.json').read_text())
m = mujoco.MjModel.from_xml_path(scene); rig = H.HandRig.build(m, 'robot_right_', 'right'); wc = np.array([rig.wrist_qpos[k] for k in ('x', 'y', 'z', 'rx', 'ry', 'rz')])
fnames = list(rig.finger_qpos); fcols = np.array([rig.finger_qpos[k] for k in fnames]); facts = np.array([rig.finger_act[k] for k in fnames])
corr = TR.load_reference(ced / 'tracking_reference.npz', m); orig = corr['qpos']; oc = int(m.joint('wooden_spoon_free').qposadr[0])
tref = TR.load_reference(TE / 'reference.npz', m); seedQ = tref['qpos'].copy()
for name in ('wooden_spoon', 'blue_cup'):
    c = int(m.joint(name + '_free').qposadr[0]); seedQ[:, c:c + 7] = orig[:, c:c + 7]
seed = TR.save_reference(out / 'seed_reference.npz', m, seedQ, tref['labels'], tref['fps'], tref['segment'], tref['segments'],
    source='v19 teacher (Stream J chainB/c2) commands; object target columns = corrected (V2a) tracking_reference', arms_coupled=False, floating_runtime=True)
p0 = orig[0, oc:oc + 3]; R0 = R.from_quat(orig[0, oc + 3:oc + 7], scalar_first=True)
tr = np.load(r['execution']['trajectory']); k = round(a.held * 20); q = tr['qpos'][k]; ctrl = tr['ctrl'][k]
po = q[oc:oc + 3]; Ro = R.from_quat(q[oc + 3:oc + 7], scalar_first=True)
local_p = Ro.inv().apply(q[wc[:3]] - po); local_R = Ro.inv() * R.from_euler('XYZ', q[wc[3:]]); closed = ctrl[facts].copy()
opened = tr['ctrl'][30][facts].copy()
def anchor(lp, lR, fingers):
    x = orig[0].copy(); x[wc[:3]] = p0 + R0.apply(lp); x[wc[3:]] = continuous_euler_xyz(R.from_matrix((R0 * lR).as_matrix()[None]), orig[0, wc[3:]])[0]; x[fcols] = fingers; return x
def power(frac, fingers=('middle', 'ring', 'pinky')):
    f = closed.copy()
    for fi in fingers:
        for jn in rig.flexion_joints(fi): lo, hi = rig.finger_range[jn]; f[fnames.index(jn)] = lo + frac * (hi - lo)
    return f
specs = []
def spec_for(name, pose, extra):
    apth = out / f'anchor_{name}.npz'; np.savez_compressed(apth, qpos=pose)
    times = dict(start=1.3, hover=1.8, contact=2.1, closed=2.35, transport=2.55, release=a.release)
    s = dict(episode='episode_000032', scene=scene, reference=str(seed), original_reference=str(ced / 'tracking_reference.npz'), source=man['source'], geometry=str(ced / 'metric_geometry.json'),
             manifest=str(ced / 'input_manifest.json'), object_name='wooden_spoon', side='right', anchor_pose=str(apth), open_fingers=opened, finger_names=fnames, times=times,
             hover_offset=[0, 0, .06], horizon_s=a.horizon, score_hold_start_s=2.35, score_track_start_s=2.6, right_hand_holds_until_end=False, teacher=str(TE), held_time_s=a.held, **extra)
    dump(out / f'{name}.json', s); specs.append(str(out / f'{name}.json'))
spec_for('pinch', anchor(local_p, local_R, closed), dict(grasp_generator=f'teacher achieved hold at {a.held} s transplanted to the resting spoon'))
for frac in (0.6, 0.85):
    spec_for(f'power{frac}', anchor(local_p, local_R, power(frac)), dict(grasp_generator=f'held + middle/ring/pinky flexion to {frac} of range'))
for dx in (-0.03, 0.03):
    spec_for(f'power0.85_x{int(dx*100):+d}', anchor(local_p + np.array([dx, 0, 0]), local_R, power(0.85)), dict(grasp_generator=f'power0.85 shifted {dx*100:+.0f} cm along the spoon axis (toward the handle end)'))
spec_for('pinch_mid0.85', anchor(local_p, local_R, power(0.85, ('middle',))), dict(grasp_generator='held + middle finger flexed to 0.85'))
json.dump(specs, open(out / 'all_specs.json', 'w'), indent=1); print(len(specs), 'specs; local_p', np.round(local_p, 3), 'rotvec', np.round(local_R.as_rotvec(), 2))
