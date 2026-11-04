"""Reference-level carry correction (iterative learning control) for any held object: in the carry window the carrying
wrist is rotated about the object's reference position by the smoothed negative orientation error and shifted by the
position error (accumulated over iterations); the correction is held through the release and blended out after.
usage: ilc_carry_generic.py --source TEACHER --out DIR --side right --object mini_dust_pan --t0 2.4 --t1 10.5 --t-clear 11.8
       --episode-dir R/episodes/episode_000038 --iterations 3 [--gain-rot 0.6 --gain-pos 0.6 --sigma-s 0.25]"""
import argparse, json, sys
from pathlib import Path
import numpy as np, mujoco
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R
ROOT = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017'); sys.path[:0] = [str(ROOT / 'code'), '/data1/SRL_V2D/third_party/DexAgent']
from tool_library.common import hand as H, tracking as TR
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files
from wrist_reference import continuous_euler_xyz
sm = lambda x: np.clip(x, 0, 1) ** 3 * (10 - 15 * np.clip(x, 0, 1) + 6 * np.clip(x, 0, 1) ** 2)
def dump(p, o): Path(p).write_text(json.dumps(o, indent=1, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)) + '\n')

def one(src, out, a, it):
    out.mkdir(parents=True, exist_ok=True); prev = json.loads((src / 'result.json').read_text()); scene = prev['execution']['scene']
    m = mujoco.MjModel.from_xml_path(scene); ref = TR.load_reference(src / 'reference.npz', m); Q0 = ref['qpos'].copy(); fps = ref['fps']; T = len(Q0); t = np.arange(T) / fps
    rig = H.HandRig.build(m, f'robot_{a.side}_', a.side); wc = np.array([rig.wrist_qpos[k] for k in ('x', 'y', 'z', 'rx', 'ry', 'rz')])
    epd = Path(a.episode_dir); orig = TR.load_reference(epd / 'tracking_reference.npz', m)['qpos']; oc = int(m.joint(a.object + '_free').qposadr[0])
    p_ref = orig[:, oc:oc + 3]; R_ref = R.from_quat(orig[:, oc + 3:oc + 7], scalar_first=True)
    tr = np.load(prev['execution']['trajectory']); i = list(tr['object_names']).index(a.object); n = min(T, len(tr['qpos']))
    p_ach = tr['object_positions'][:n, i]; R_ach = R.from_quat(tr['object_quaternions_wxyz'][:n, i][:, [1, 2, 3, 0]])
    acc = prev.get('ilc', {}); rot = np.asarray(acc.get('rotation_table', np.zeros((T, 3)))); pos = np.asarray(acc.get('translation_table', np.zeros((T, 3))))
    base = np.load(acc['base_qpos'])['qpos'] if 'base_qpos' in acc else Q0.copy()
    sig = a.sigma_s * fps; er = np.zeros((T, 3)); ep = np.zeros((T, 3))
    er[:n] = gaussian_filter1d((R_ach * R_ref[:n].inv()).as_rotvec(), sig, axis=0, mode='nearest'); ep[:n] = gaussian_filter1d(p_ref[:n] - p_ach, sig, axis=0, mode='nearest')
    win = (t >= a.t0) & (t <= a.t1)
    rot[win] = (R.from_rotvec(-a.gain_rot * er[win]) * R.from_rotvec(rot[win])).as_rotvec(); pos[win] = pos[win] + a.gain_pos * ep[win]
    last = np.flatnonzero(win)[-1]; rot[last + 1:] = rot[last]; pos[last + 1:] = pos[last]; pos[:np.flatnonzero(win)[0]] = 0; rot[:np.flatnonzero(win)[0]] = 0
    gate = sm((t - a.t0) / 0.3) * (1 - sm((t - a.t_clear) / 0.5)); Q = base.copy()
    for k in range(T):
        g = gate[k]
        if g <= 0: continue
        p = Q[k, wc[:3]].copy(); ang = Q[k, wc[3:]].copy(); Rw = R.from_euler('XYZ', ang); Rc = R.from_rotvec(g * rot[k]); c = p_ref[k]
        Q[k, wc[:3]] = c + Rc.apply(p - c) + g * pos[k]; Q[k, wc[3:]] = continuous_euler_xyz(R.from_matrix((Rc * Rw).as_matrix()[None]), ang)[0]
    Q[0] = Q0[0]; lim = m.jnt_range[m.actuator_trnid[:, 0]]; qa = m.jnt_qposadr[m.actuator_trnid[:, 0]]; Q[:, qa] = np.clip(Q[:, qa], lim[:, 0], lim[:, 1])
    refp = TR.save_reference(out / 'reference.npz', m, Q, ref['labels'], fps, ref['segment'], ref['segments'], source='carry correction (reference-level ILC) on the carrying wrist; immutable object targets/clock', arms_coupled=False, floating_runtime=True)
    bp = out / 'base_qpos.npz'; np.savez_compressed(bp, qpos=base)
    ex = execute_reference(scene, refp, out / 'execution'); man = json.loads((epd / 'input_manifest.json').read_text())
    met = evaluate_files(man['source'], ex['trajectory'], epd / 'input_manifest.json', ex['scene'], epd / 'metric_geometry.json')
    res = dict(prev, reference=str(refp), execution=ex, metrics=met, ilc=dict(iteration=int(acc.get('iteration', 0)) + 1, rotation_table=rot, translation_table=pos, base_qpos=str(bp), window=[a.t0, a.t1], t_clear=a.t_clear, gain_rot=a.gain_rot, gain_pos=a.gain_pos), source=str(src))
    dump(out / 'result.json', res)
    o = met['official_style_reference_proxy']['objects'][a.object]
    tr2 = np.load(ex['trajectory']); Ra2 = R.from_quat(tr2['object_quaternions_wxyz'][:n, i][:, [1, 2, 3, 0]]); e2 = np.degrees((Ra2 * R_ref[:n].inv()).magnitude())
    print(json.dumps(dict(iteration=it, mp_sr=met['reference_proxy_mp_sr'], cm=round(o['mean_position_error_cm'][0], 2), deg=round(o['mean_orientation_error_degrees'][0], 1), carry_deg=round(float(e2[int(a.t0*fps):int(a.t1*fps)].mean()), 1), completed=ex['execution_completed'])), flush=True)
    return out

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for k in ('source', 'out', 'side', 'object', 'episode_dir'): p.add_argument('--' + k.replace('_', '-'), required=True)
    for k, dv in (('t0', 2.4), ('t1', 10.5), ('t_clear', 11.8), ('gain_rot', .6), ('gain_pos', .6), ('sigma_s', .25)): p.add_argument('--' + k.replace('_', '-'), type=float, default=dv)
    p.add_argument('--iterations', type=int, default=3); a = p.parse_args(); src = Path(a.source)
    for k in range(a.iterations): src = one(src, Path(a.out) / f'iter{k}', a, k)
