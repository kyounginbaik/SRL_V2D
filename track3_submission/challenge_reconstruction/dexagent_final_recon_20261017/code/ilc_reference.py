"""Reference-level carry correction for a power-grasp pickup plan (left hand on the white pot).

Starting from a physics_pickup_search result (reference.npz + full-horizon execution), each iteration:
  1. measures the pot's per-frame pose error (achieved vs immutable reference) in the carry window,
  2. rotates/translates the LEFT wrist commands against it (rotation about the pot's reference centre),
  3. rewrites the release: hold the final carry pose, open the fingers, retreat upward, blend to the old plan,
  4. executes the new reference over the full original horizon and scores it (same scorer as the search).
Object targets, timing, scene and the right-hand plan are never changed.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, mujoco
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R
B = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(B.parents[1] / 'third_party/DexAgent')); sys.path.insert(0, str(B / 'code'))
from tool_library.common import hand as H, tracking as TR
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files
from wrist_reference import continuous_euler_xyz
from release_feedback_policy import smooth


def dump(p, o): Path(p).write_text(json.dumps(o, indent=1, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)) + '\n')


def run(source, out, gain_rot, gain_pos, sigma_s, open_s, retreat_m, retreat_s, blend_s, no_ilc, press_m=0.0, press_s=0.3, no_blend=False):
    source, out = Path(source), Path(out); out.mkdir(parents=True, exist_ok=True)
    prev = json.loads((source / 'result.json').read_text()); spec = prev['spec']
    m = mujoco.MjModel.from_xml_path(prev['execution']['scene']); ref = TR.load_reference(source / 'reference.npz', m); Q0 = ref['qpos'].copy(); fps = ref['fps']; T = len(Q0); t = np.arange(T) / fps
    rig = H.HandRig.build(m, 'robot_left_', 'left'); wc = np.array([rig.wrist_qpos[k] for k in ('x', 'y', 'z', 'rx', 'ry', 'rz')]); fc = np.array(list(rig.finger_qpos.values()))
    original = TR.load_reference(spec['original_reference'], m)['qpos']; oc = int(m.joint('white_pot_free').qposadr[0])
    p_ref = original[:, oc:oc + 3]; R_ref = R.from_quat(original[:, oc + 3:oc + 7], scalar_first=True)
    tr = np.load(prev['execution']['trajectory']); i = list(tr['object_names']).index('white_pot'); n = min(T, len(tr['qpos']))
    p_ach = tr['object_positions'][:n, i]; R_ach = R.from_quat(tr['object_quaternions_wxyz'][:n, i][:, [1, 2, 3, 0]])
    ts = spec['times']; timing = prev.get('best', {}).get('timing', {}); t_closed = float(timing.get('closed', ts['closed'])); t_rel = float(ts['release'])
    # accumulated correction tables (ungated), carried across iterations
    acc = prev.get('ilc', {}); rot_tab = np.asarray(acc.get('rotation_table', np.zeros((T, 3)))); pos_tab = np.asarray(acc.get('translation_table', np.zeros((T, 3))))
    # the plan we modify is the un-released, un-corrected pickup plan (kept from the first iteration)
    base = np.asarray(acc.get('base_qpos', Q0)) if 'base_qpos' in acc else Q0.copy()
    if 'base_qpos' in acc: base = np.load(acc['base_qpos'])['qpos']
    sig = sigma_s * fps; err_rot = np.zeros((T, 3)); err_pos = np.zeros((T, 3))
    err_rot[:n] = gaussian_filter1d((R_ach * R_ref[:n].inv()).as_rotvec(), sig, axis=0, mode='nearest'); err_pos[:n] = gaussian_filter1d(p_ref[:n] - p_ach, sig, axis=0, mode='nearest')
    win = (t >= t_closed) & (t <= t_rel)
    if not no_ilc:
        rot_tab[win] = (R.from_rotvec(-gain_rot * err_rot[win]) * R.from_rotvec(rot_tab[win])).as_rotvec(); pos_tab[win] = pos_tab[win] + gain_pos * err_pos[win]
    # hold the last correction through the release (hand still on the pot)
    last = np.flatnonzero(win)[-1]; rot_tab[last + 1:] = rot_tab[last]; pos_tab[last + 1:] = pos_tab[last]
    gate = smooth((t - t_closed) / 0.3); Q = base.copy()
    for k in range(T):
        g = gate[k]
        if g <= 0: continue
        pos = Q[k, wc[:3]].copy(); ang = Q[k, wc[3:]].copy(); Rw = R.from_euler('XYZ', ang); Rc = R.from_rotvec(g * rot_tab[k]); c = p_ref[k]
        Q[k, wc[:3]] = c + Rc.apply(pos - c) + g * pos_tab[k]; Q[k, wc[3:]] = continuous_euler_xyz(R.from_matrix((Rc * Rw).as_matrix()[None]), ang)[0]
    # release rewrite: hold pose, open, retreat, blend back to the old plan
    f_rel = int(round(t_rel * fps)); hold_pos = Q[f_rel - 1, wc[:3]].copy(); hold_ang = Q[f_rel - 1, wc[3:]].copy(); closed = Q[f_rel - 1, fc].copy(); opened = np.asarray(spec['open_fingers'], float)
    for k in range(f_rel, T):
        dt = t[k] - t_rel; pr_ = smooth(dt / press_s); o = smooth((dt - press_s) / open_s); r = smooth((dt - press_s - open_s) / retreat_s)
        bl = 0. if no_blend else smooth((dt - press_s - open_s - retreat_s - 0.2) / blend_s)
        wrist_p = hold_pos + np.array([0., 0., -press_m * pr_ + retreat_m * r]); wrist_a = hold_ang; fing = (1 - o) * closed + o * opened
        Q[k, wc[:3]] = (1 - bl) * wrist_p + bl * Q0[k, wc[:3]]; Q[k, fc] = (1 - bl) * fing + bl * Q0[k, fc]
        Ra, Rb = R.from_euler('XYZ', wrist_a), R.from_euler('XYZ', Q0[k, wc[3:]]); Rk = Ra * R.from_rotvec((Ra.inv() * Rb).as_rotvec() * bl)
        Q[k, wc[3:]] = continuous_euler_xyz(R.from_matrix(Rk.as_matrix()[None]), Q[k - 1, wc[3:]])[0]
    Q[0] = Q0[0]; lim = m.jnt_range[m.actuator_trnid[:, 0]]; qadr = m.jnt_qposadr[m.actuator_trnid[:, 0]]; Q[:, qadr] = np.clip(Q[:, qadr], lim[:, 0], lim[:, 1])
    refpath = TR.save_reference(out / 'reference.npz', m, Q, ref['labels'], fps, ref['segment'], ref['segments'], source='Power-grasp pickup plan + reference-level carry correction and rewritten release; immutable object targets/clock', arms_coupled=False, floating_runtime=True)
    base_path = out / 'base_qpos.npz'; np.savez_compressed(base_path, qpos=base)
    execution = execute_reference(prev['execution']['scene'], refpath, out / 'execution')
    metrics = evaluate_files(spec['source'], execution['trajectory'], spec['manifest'], execution['scene'], spec['geometry'])
    objs = {k: (round(v['mean_position_error_cm'][0], 2), round(v['mean_orientation_error_degrees'][0], 1)) for k, v in metrics['official_style_reference_proxy']['objects'].items()}
    result = dict(prev, reference=str(refpath), execution=execution, metrics=metrics, ilc=dict(iteration=int(acc.get('iteration', 0)) + 1, rotation_table=rot_tab, translation_table=pos_tab, base_qpos=str(base_path),
                  gain_rot=gain_rot, gain_pos=gain_pos, sigma_s=sigma_s, window=[t_closed, t_rel], release=dict(open_s=open_s, retreat_m=retreat_m, retreat_s=retreat_s, blend_s=blend_s, press_m=press_m, press_s=press_s, no_blend=no_blend),
                  residual_rot_deg_window=float(np.degrees(np.linalg.norm(err_rot[win], axis=1)).mean()), residual_pos_cm_window=float(100 * np.linalg.norm(err_pos[win], axis=1).mean())), source=str(source))
    dump(out / 'result.json', result)
    print(json.dumps(dict(iteration=result['ilc']['iteration'], mp_sr=metrics['official_style_reference_proxy']['mp_sr'], objects=objs, residual_rot=round(result['ilc']['residual_rot_deg_window'], 1), residual_pos=round(result['ilc']['residual_pos_cm_window'], 2), completed=execution['execution_completed'])), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--source', required=True); ap.add_argument('--out', required=True); ap.add_argument('--iterations', type=int, default=1)
    ap.add_argument('--gain-rot', type=float, default=0.6); ap.add_argument('--gain-pos', type=float, default=0.6); ap.add_argument('--sigma-s', type=float, default=0.25)
    ap.add_argument('--open-s', type=float, default=0.4); ap.add_argument('--retreat-m', type=float, default=0.08); ap.add_argument('--retreat-s', type=float, default=0.6); ap.add_argument('--blend-s', type=float, default=1.0)
    ap.add_argument('--no-ilc', action='store_true', help='only rewrite the release (iteration 0)'); ap.add_argument('--press-m', type=float, default=0.0); ap.add_argument('--press-s', type=float, default=0.3)
    ap.add_argument('--no-blend', action='store_true', help='hold the retreated pose to the end instead of blending back to the old plan')
    a = ap.parse_args(); src = a.source
    for k in range(a.iterations):
        d = Path(a.out) / f'iter{k}'; run(src, d, a.gain_rot, a.gain_pos, a.sigma_s, a.open_s, a.retreat_m, a.retreat_s, a.blend_s, a.no_ilc and k == 0, a.press_m, a.press_s, a.no_blend); src = d
