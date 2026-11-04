"""Rewrite the pot-hand release of a finished plan (reference.npz) and execute/score it over the full horizon.
Release schedule after t_rel: press down (press_m over press_s) -> open fingers to `open_frac` of the way over open_s
-> retreat `retreat_m` along `retreat_dir` ('up' or 'handle' = the pot's handle axis away from the body) over retreat_s
-> finish opening -> hold to the end. Object targets, timing, scene and the right-hand plan are unchanged.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, mujoco
from scipy.spatial.transform import Rotation as R
B = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(B.parents[1] / 'third_party/DexAgent')); sys.path.insert(0, str(B / 'code'))
from tool_library.common import hand as H, tracking as TR
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files
from release_feedback_policy import smooth


def dump(p, o): Path(p).write_text(json.dumps(o, indent=1, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)) + '\n')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--source', required=True, help='dir with reference.npz + result.json (execution.scene)'); ap.add_argument('--spec', required=True, help='result.json of the pickup/ILC chain (spec with times/open_fingers)')
    ap.add_argument('--out', required=True); ap.add_argument('--press-m', type=float, default=0.01); ap.add_argument('--press-s', type=float, default=0.3); ap.add_argument('--open-s', type=float, default=0.8)
    ap.add_argument('--open-frac', type=float, default=1.0, help='fraction opened before the retreat starts'); ap.add_argument('--retreat-dir', choices=['up', 'handle', 'handle_up'], default='up')
    ap.add_argument('--retreat-m', type=float, default=0.08); ap.add_argument('--retreat-s', type=float, default=0.6); ap.add_argument('--finish-s', type=float, default=0.4)
    a = ap.parse_args(); src = Path(a.source); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    prev = json.loads((src / 'result.json').read_text()); spec = json.loads(Path(a.spec).read_text())['spec']; scene = prev['execution']['scene']
    m = mujoco.MjModel.from_xml_path(scene); ref = TR.load_reference(src / 'reference.npz', m); Q0 = ref['qpos'].copy(); Q = Q0.copy(); fps = ref['fps']; T = len(Q); t = np.arange(T) / fps
    rig = H.HandRig.build(m, 'robot_left_', 'left'); wc = np.array([rig.wrist_qpos[k] for k in ('x', 'y', 'z', 'rx', 'ry', 'rz')]); fc = np.array(list(rig.finger_qpos.values()))
    original = TR.load_reference(spec['original_reference'], m)['qpos']; oc = int(m.joint('white_pot_free').qposadr[0])
    t_rel = float(spec['times']['release']); f_rel = int(round(t_rel * fps))
    hold_pos = Q[f_rel - 1, wc[:3]].copy(); hold_ang = Q[f_rel - 1, wc[3:]].copy(); closed = Q[f_rel - 1, fc].copy(); opened = np.asarray(spec['open_fingers'], float)
    Rpot = R.from_quat(original[f_rel, oc + 3:oc + 7], scalar_first=True); handle = Rpot.apply([0, 0, 1.]); handle /= np.linalg.norm(handle)
    up = np.array([0, 0, 1.]); direction = {'up': up, 'handle': handle, 'handle_up': (handle + 0.5 * up) / np.linalg.norm(handle + 0.5 * up)}[a.retreat_dir]
    for k in range(f_rel, T):
        dt = t[k] - t_rel; pr_ = smooth(dt / a.press_s); o1 = a.open_frac * smooth((dt - a.press_s) / a.open_s); r = smooth((dt - a.press_s - a.open_s) / a.retreat_s)
        o2 = (1 - a.open_frac) * smooth((dt - a.press_s - a.open_s - a.retreat_s) / a.finish_s); o = o1 + o2
        Q[k, wc[:3]] = hold_pos - up * a.press_m * pr_ + direction * a.retreat_m * r; Q[k, wc[3:]] = hold_ang; Q[k, fc] = (1 - o) * closed + o * opened
    lim = m.jnt_range[m.actuator_trnid[:, 0]]; qadr = m.jnt_qposadr[m.actuator_trnid[:, 0]]; Q[:, qadr] = np.clip(Q[:, qadr], lim[:, 0], lim[:, 1]); Q[0] = Q0[0]
    refpath = TR.save_reference(out / 'reference.npz', m, Q, ref['labels'], fps, ref['segment'], ref['segments'], source='Finished pickup/carry/spoon plan with rewritten pot release; immutable object targets/clock', arms_coupled=False, floating_runtime=True)
    execution = execute_reference(scene, refpath, out / 'execution'); metrics = evaluate_files(spec['source'], execution['trajectory'], spec['manifest'], execution['scene'], spec['geometry'])
    objs = {k: (round(v['mean_position_error_cm'][0], 2), round(v['mean_orientation_error_degrees'][0], 1)) for k, v in metrics['official_style_reference_proxy']['objects'].items()}
    tr = np.load(execution['trajectory']); j = list(tr['object_names']).index('white_pot'); pa = tr['object_positions'][:, j]; pr = original[:, oc:oc + 3]
    after = np.linalg.norm(pa[f_rel + 20:] - pr[f_rel + 20:], axis=1) * 100
    result = dict(reference=str(refpath), execution=execution, metrics=metrics, spec=spec, source=str(src), release=vars(a), pot_after_release_mean_cm=float(after.mean()), pot_final_cm=float(after[-1]))
    dump(out / 'result.json', result); print(json.dumps(dict(out=out.name, mp_sr=metrics['official_style_reference_proxy']['mp_sr'], objects=objs, pot_after_release_cm=round(float(after.mean()), 2), pot_final_cm=round(float(after[-1]), 2))))


if __name__ == '__main__':
    main()
