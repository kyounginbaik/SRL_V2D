"""Stream P ep33 phase 6 (compose world): power-grasp pickup specs for the unchanged physics_pickup_search_v2 (pattern: Stream R
streamR_ep32/code/prepare_pickup32.py). Scene = compose world staging_C/scene (objects start at compose frame 0); spoon/cup target
columns = staging_C/episode_compose tracking_reference (compose tracks in our frames); seed reference = the v21 ep33 teacher
(refinement_20261021_ep33/search/v12_segE/c2_R_9.8_12.3) commands, with the LEFT-hand columns replaced by the commands of the solved
compose-world cup re-target actor (C/t3_L2_r0._L1.6,2.4,20,21,1,1: cup 0.83 cm / 8.8 deg vs compose). Anchors = the teacher's achieved
right-hand hold at --held s transplanted onto the resting compose spoon: pinch, power wrap (middle/ring/pinky flexion to 0.6 / 0.85),
power0.85 shifted +-3 cm along the handle, pinch + middle. After closure the wrist follows (compose spoon target) o (grasp offset).
usage: prepare_pickup33.py OUT_DIR [--held 3.5] [--release 14.0] [--horizon 5.5]"""
import json, sys, argparse
from pathlib import Path
import numpy as np, mujoco
from scipy.spatial.transform import Rotation as R
LEG = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017'); sys.path[:0] = ['/data1/SRL_V2D/third_party/DexAgent', str(LEG / 'code')]
from tool_library.common import hand as H, tracking as TR
from wrist_reference import continuous_euler_xyz
from physics_pickup_search_v2 import dump
P = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamP_ep33'); CW = P / 'staging_C'; ced = CW / 'episode_compose/episode_000033'; SC = CW / 'scene/scene_physics.xml'
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--held', type=float, default=3.5); ap.add_argument('--release', type=float, default=14.0); ap.add_argument('--horizon', type=float, default=5.5)
ap.add_argument('--teacher', default=str(LEG / 'refinement_20261021_ep33/search/v12_segE/c2_R_9.8_12.3')); ap.add_argument('--rest-rollout', default=str(P / 'C/port_cuptilt/evaluation/execution/trajectory.npz')); ap.add_argument('--rest-frame', type=int, default=34)
ap.add_argument('--left-actor', default=str(P / 'C/t3_L2_r0._L1.6,2.4,20,21,1,1'))
a = ap.parse_args(); out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True); TE = Path(a.teacher)
r = json.loads((TE / 'result.json').read_text()); man = json.loads((ced / 'input_manifest.json').read_text())
m = mujoco.MjModel.from_xml_path(str(SC)); rig = H.HandRig.build(m, 'robot_right_', 'right'); wc = np.array([rig.wrist_qpos[k] for k in ('x', 'y', 'z', 'rx', 'ry', 'rz')])
fnames = list(rig.finger_qpos); fcols = np.array([rig.finger_qpos[k] for k in fnames]); facts = np.array([rig.finger_act[k] for k in fnames])
corr = TR.load_reference(ced / 'tracking_reference.npz', m); orig = corr['qpos']; oc = int(m.joint('wooden_spoon_free').qposadr[0])
tref = TR.load_reference(TE / 'reference.npz', m); seedQ = tref['qpos'].copy()
for name in ('wooden_spoon', 'blue_cup'):
    c = int(m.joint(name + '_free').qposadr[0]); seedQ[:, c:c + 7] = orig[:, c:c + 7]
la = json.loads((Path(a.left_actor) / 'evaluation/result.json').read_text()); lt = np.load(la['execution']['trajectory'])
left_acts = [i for i in range(m.nu) if (m.actuator(i).name or '').startswith('robot_left_')]
for i in left_acts: seedQ[:, m.jnt_qposadr[m.actuator_trnid[i, 0]]] = lt['ctrl'][:len(seedQ), i]
seed = TR.save_reference(out / 'seed_reference.npz', m, seedQ, tref['labels'], tref['fps'], tref['segment'], tref['segments'],
    source='v21 ep33 teacher commands (right); left = commands of the compose cup re-target actor ' + a.left_actor + '; object target columns = compose (staging_C) tracking_reference', arms_coupled=False, floating_runtime=True)
p0 = orig[0, oc:oc + 3]; R0 = R.from_quat(orig[0, oc + 3:oc + 7], scalar_first=True)
if a.rest_rollout:   # anchor on the SETTLED spoon (the compose frame-0 spoon is rolled ~17 deg and settles flat)
    rt = np.load(a.rest_rollout); ri = [str(x) for x in rt['object_names']].index('wooden_spoon'); p0 = rt['object_positions'][a.rest_frame, ri]; R0 = R.from_quat(rt['object_quaternions_wxyz'][a.rest_frame, ri][[1, 2, 3, 0]])
tr = np.load(r['execution']['trajectory']); k = round(a.held * 20); q = tr['qpos'][k]; ctrl = tr['ctrl'][k]
po = q[oc:oc + 3]; Ro = R.from_quat(q[oc + 3:oc + 7], scalar_first=True)
local_p = Ro.inv().apply(q[wc[:3]] - po); local_R = Ro.inv() * R.from_euler('XYZ', q[wc[3:]]); closed = ctrl[facts].copy(); opened = tr['ctrl'][30][facts].copy()
def anchor(lp, lR, fingers):
    x = orig[0].copy(); x[wc[:3]] = p0 + R0.apply(lp); x[wc[3:]] = continuous_euler_xyz(R.from_matrix((R0 * lR).as_matrix()[None]), orig[0, wc[3:]])[0]; x[fcols] = fingers; return x
def power(frac, fingers=('middle', 'ring', 'pinky')):
    f = closed.copy()
    for fi in fingers:
        for jn in rig.flexion_joints(fi): lo, hi = rig.finger_range[jn]; f[fnames.index(jn)] = lo + frac * (hi - lo)
    return f
specs = []
REST = dict(rest_rollout=a.rest_rollout, rest_frame=a.rest_frame)
def spec_for(name, pose, extra):
    apth = out / f'anchor_{name}.npz'; np.savez_compressed(apth, qpos=pose)
    times = dict(start=1.5, hover=1.95, contact=2.2, closed=2.45, transport=2.7, release=a.release)
    s = dict(episode='episode_000033', scene=str(SC), reference=str(seed), original_reference=str(ced / 'tracking_reference.npz'), source=man['source'], geometry=str(ced / 'metric_geometry.json'),
             manifest=str(ced / 'input_manifest.json'), object_name='wooden_spoon', side='right', anchor_pose=str(apth), open_fingers=opened, finger_names=fnames, times=times,
             hover_offset=[0, 0, .06], horizon_s=a.horizon, score_hold_start_s=2.45, score_track_start_s=2.75, right_hand_holds_until_end=False, teacher=str(TE), held_time_s=a.held, anchor_on=REST, **extra)
    dump(out / f'{name}.json', s); specs.append(str(out / f'{name}.json'))
spec_for('pinch', anchor(local_p, local_R, closed), dict(grasp_generator=f'teacher achieved hold at {a.held} s transplanted to the resting compose spoon'))
for frac in (0.6, 0.85):
    spec_for(f'power{frac}', anchor(local_p, local_R, power(frac)), dict(grasp_generator=f'held + middle/ring/pinky flexion to {frac} of range'))
for dx in (-0.03, 0.03):
    spec_for(f'power0.85_x{int(round(dx*100)):+d}', anchor(local_p + np.array([dx, 0, 0]), local_R, power(0.85)), dict(grasp_generator=f'power0.85 shifted {dx*100:+.0f} cm along the spoon axis'))
spec_for('pinch_mid0.85', anchor(local_p, local_R, power(0.85, ('middle',))), dict(grasp_generator='held + middle finger flexed to 0.85'))
json.dump(specs, open(out / 'all_specs.json', 'w'), indent=1); print(len(specs), 'specs; local_p', np.round(local_p, 3), 'rotvec', np.round(local_R.as_rotvec(), 2))
