"""Stream B v22 copy of streamR prepare_pickup32c.py (episode name from the manifest). Original: Stream R ep32 phase 6: grasp specs (physics_pickup_search_v2) for either hand in the compose-retargeted world. After closure the wrist follows
(target object pose of the episode dir's tracking_reference) o (grasp offset). Seed = SEED_REF (a reference.npz) with its object columns replaced by the
episode dir's targets. Anchor = the TEACHER rollout's achieved hold of SIDE on OBJECT at HELD s re-expressed on the object's frame-0 target, closed fingers =
commanded fingers at HELD; variants: power (non-thumb flexion to FRAC of range). Open fingers = commanded at OPEN_T.
usage: prepare_pickup32c.py OUT_DIR EPISODE_DIR SCENE --side left|right --object NAME --seed-ref REF.npz --teacher DIR --held S --open-t S
        --times start,hover,contact,closed,transport,release [--fracs 0.6,0.85] [--horizon 12.9] [--hold-start S] [--track-start S]"""
import json, sys, argparse
from pathlib import Path
import numpy as np, mujoco
from scipy.spatial.transform import Rotation as R
LEG = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017'); sys.path[:0] = ['/data1/SRL_V2D/third_party/DexAgent', str(LEG / 'code')]
from tool_library.common import hand as H, tracking as TR
from wrist_reference import continuous_euler_xyz
from physics_pickup_search_v2 import dump
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('epd'); ap.add_argument('scene'); ap.add_argument('--side', required=True); ap.add_argument('--object', required=True)
ap.add_argument('--seed-ref', required=True); ap.add_argument('--teacher', required=True); ap.add_argument('--held', type=float, required=True); ap.add_argument('--open-t', type=float, required=True)
ap.add_argument('--times', required=True); ap.add_argument('--fracs', default='0.6,0.85'); ap.add_argument('--horizon', type=float, default=12.9); ap.add_argument('--hold-start', type=float); ap.add_argument('--track-start', type=float); a = ap.parse_args()
out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True); epd = Path(a.epd).resolve(); TE = Path(a.teacher); side = a.side; obj = a.object
man = json.loads((epd / 'input_manifest.json').read_text()); m = mujoco.MjModel.from_xml_path(a.scene); rig = H.HandRig.build(m, f'robot_{side}_', side)
wc = np.array([rig.wrist_qpos[k] for k in ('x', 'y', 'z', 'rx', 'ry', 'rz')]); fnames = list(rig.finger_qpos); fcols = np.array([rig.finger_qpos[k] for k in fnames]); facts = np.array([rig.finger_act[k] for k in fnames])
orig = TR.load_reference(epd / 'tracking_reference.npz', m)['qpos']; oc = int(m.joint(obj + '_free').qposadr[0])
sref = TR.load_reference(a.seed_ref, m); seedQ = sref['qpos'].copy()
for name in ('wooden_spoon', 'blue_cup'):
    c = int(m.joint(name + '_free').qposadr[0]); seedQ[:, c:c + 7] = orig[:, c:c + 7]
seed = TR.save_reference(out / 'seed_reference.npz', m, seedQ, sref['labels'], sref['fps'], sref['segment'], sref['segments'], source=f'seed {a.seed_ref}; object target columns = {epd}/tracking_reference', arms_coupled=False, floating_runtime=True)
p0 = orig[0, oc:oc + 3]; R0 = R.from_quat(orig[0, oc + 3:oc + 7], scalar_first=True)
r = json.loads((TE / 'result.json').read_text()); tr = np.load(r['execution']['trajectory']); k = round(a.held * 20); q = tr['qpos'][k]; ctrl = tr['ctrl'][k]
po = q[oc:oc + 3]; Ro = R.from_quat(q[oc + 3:oc + 7], scalar_first=True)
local_p = Ro.inv().apply(q[wc[:3]] - po); local_R = Ro.inv() * R.from_euler('XYZ', q[wc[3:]]); closed = ctrl[facts].copy(); opened = tr['ctrl'][round(a.open_t * 20)][facts].copy()
def anchor(fingers):
    x = orig[0].copy(); x[wc[:3]] = p0 + R0.apply(local_p); x[wc[3:]] = continuous_euler_xyz(R.from_matrix((R0 * local_R).as_matrix()[None]), orig[0, wc[3:]])[0]; x[fcols] = fingers; return x
def power(frac):
    f = closed.copy()
    for fi in ('middle', 'ring', 'pinky'):
        for jn in rig.flexion_joints(fi): lo, hi = rig.finger_range[jn]; f[fnames.index(jn)] = lo + frac * (hi - lo)
    return f
t = [float(x) for x in a.times.split(',')]; times = dict(start=t[0], hover=t[1], contact=t[2], closed=t[3], transport=t[4], release=t[5]); specs = []
def spec_for(name, pose, gen):
    apth = out / f'anchor_{name}.npz'; np.savez_compressed(apth, qpos=pose)
    s = dict(episode=man['episode'], scene=a.scene, reference=str(seed), original_reference=str(epd / 'tracking_reference.npz'), source=man['source'], geometry=str(epd / 'metric_geometry.json'),
             manifest=str(epd / 'input_manifest.json'), object_name=obj, side=side, anchor_pose=str(apth), open_fingers=opened, finger_names=fnames, times=times, hover_offset=[0, 0, .04 if side == 'left' else .06],
             horizon_s=a.horizon, score_hold_start_s=a.hold_start or t[3], score_track_start_s=a.track_start or t[4] + 0.05, right_hand_holds_until_end=False, teacher=str(TE), held_time_s=a.held, grasp_generator=gen)
    dump(out / f'{name}.json', s); specs.append(str(out / f'{name}.json'))
spec_for('held', anchor(closed), f'{side} teacher hold at {a.held} s on {obj} transplanted to the frame-0 target')
for fr in [float(x) for x in a.fracs.split(',') if x]: spec_for(f'power{fr}', anchor(power(fr)), f'held + middle/ring/pinky flexion to {fr}')
json.dump(specs, open(out / 'all_specs.json', 'w'), indent=1); print(len(specs), 'specs', side, obj, 'local_p', np.round(local_p, 3))
