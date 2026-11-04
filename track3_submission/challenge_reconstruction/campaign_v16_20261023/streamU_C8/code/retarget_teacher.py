"""Port an open-loop servo teacher (reference.npz of a result.json) into the compose-target world: wrists are first carried by
the setup change T = S_new @ inv(S_old), then moved with the object each hand handles: wrist_new(t) = D(t) @ T @ wrist_old(t),
D(t) = new_target(t) @ inv(T @ old_target(t)) (targets = tracking_reference object poses of each episode dir), ramped in from
identity over --ramp-s from t=0 (frame 0 = the new home). Fingers unchanged. Scene = _freeze(new stage2/scene.xml) + controller_scene
(variant). Output: reference.npz, execution/, result.json (ILC / retention / distill compatible).
usage: retarget_teacher.py --teacher DIR --old-epdir D --new-epdir D --out DIR [--assign right=beige_cup,left=white_pot] [--ramp-s 0.75]"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import argparse, json, sys, hashlib
from pathlib import Path
import numpy as np, mujoco
from scipy.spatial.transform import Rotation as R
R0 = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017'); sys.path[:0] = [str(R0 / 'code'), '/data1/SRL_V2D/third_party/DexAgent']
from floating_runtime import _freeze, execute_reference
from calibrate_controller import controller_scene
from evaluate_mp_sr import evaluate_files
from wrist_reference import continuous_euler_xyz
from tool_library.common import tracking as TR
from tool_library.common import episode as E
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
W = ['pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z']
sm = lambda x: np.clip(x, 0, 1) ** 3 * (10 - 15 * np.clip(x, 0, 1) + 6 * np.clip(x, 0, 1) ** 2)
def obj_T(Q, m, name):
    c = int(m.joint(name + '_free').qposadr[0]); T = np.tile(np.eye(4), (len(Q), 1, 1)); T[:, :3, 3] = Q[:, c:c + 3]; T[:, :3, :3] = R.from_quat(Q[:, c + 3:c + 7], scalar_first=True).as_matrix(); return T
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--teacher', required=True); ap.add_argument('--old-epdir', required=True); ap.add_argument('--new-epdir', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--assign', default='right=beige_cup,left=white_pot'); ap.add_argument('--ramp-s', type=float, default=.75); ap.add_argument('--variant', default='kp5000_r200_d5'); ap.add_argument('--scene'); ap.add_argument('--smooth-s', default='0', help='per-side sigma (s) of the delta low-pass, e.g. right=0.4,left=0'); ap.add_argument('--rot-only', action='store_true', help='low-pass only the delta rotation (object-centre path exact)'); ap.add_argument('--rot-gain', default='1', help='per-side scale of the delta rotation, e.g. right=0.6')
    a = ap.parse_args(); out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True); old, new = Path(a.old_epdir), Path(a.new_epdir); teacher = Path(a.teacher)
    S_old = np.array(json.loads((old / 'input_manifest.json').read_text())['setup_from_source']); S_new = np.array(json.loads((new / 'input_manifest.json').read_text())['setup_from_source'])
    Tch = S_new @ np.linalg.inv(S_old)
    if a.scene: scene = Path(a.scene)
    else:
        (out / 'scene').mkdir(parents=True, exist_ok=True); _, frozen = _freeze(new / 'stage2/scene.xml', out / 'scene'); scene = out / 'scene/controller_scene.xml'; controller_scene(frozen, scene, a.variant)
    m = mujoco.MjModel.from_xml_path(str(scene)); names = E.joint_names(m)
    tref = TR.load_reference(teacher / 'reference.npz', m); Q = tref['qpos'].copy(); fps = tref['fps']; Tn = len(Q); t = np.arange(Tn) / fps
    Qold = TR.load_reference(old / 'tracking_reference.npz', m)['qpos']; Qnew = TR.load_reference(new / 'tracking_reference.npz', m)['qpos']
    ramp = sm(t / a.ramp_s) if a.ramp_s > 0 else np.ones(Tn); report = {}
    for pair in a.assign.split(','):
        side, obj = pair.split('='); cols = [names.index(f'robot_{side}_{side}_{w}') for w in W]
        n = min(Tn, len(Qold), len(Qnew)); Told = obj_T(Qold[:n], m, obj); Tnew = obj_T(Qnew[:n], m, obj)
        D = np.einsum('tij,tjk->tik', Tnew, np.linalg.inv(np.einsum('ij,tjk->tik', Tch, Told)))
        D = np.concatenate([D, np.repeat(D[-1:], Tn - n, 0)]) if Tn > n else D[:Tn]
        Wt = np.tile(np.eye(4), (Tn, 1, 1)); Wt[:, :3, 3] = Q[:, cols[:3]]; Wt[:, :3, :3] = R.from_euler('XYZ', Q[:, cols[3:]]).as_matrix()
        Wt = np.einsum('ij,tjk->tik', Tch, Wt)
        sg = dict(x.split('=') for x in a.smooth_s.split(',')) if '=' in a.smooth_s else {}; sig = float(sg.get(side, 0)) * fps
        rg = dict(x.split('=') for x in a.rot_gain.split(',')) if '=' in a.rot_gain else {}; gain = float(rg.get(side, 1))
        if sig > 0 or gain != 1:
            from scipy.ndimage import gaussian_filter1d
            rvD = R.from_matrix(D[:, :3, :3]).as_rotvec(); cD = np.einsum('tij,tj->ti', D[:, :3, :3], np.einsum('ij,tj->ti', Tch[:3, :3], (Told[:, :3, 3] if Tn <= n else np.concatenate([Told[:, :3, 3], np.repeat(Told[-1:, :3, 3], Tn - n, 0)]))[:Tn]) + Tch[:3, 3]) + D[:, :3, 3]
            if sig > 0: rvD = gaussian_filter1d(rvD, sig, axis=0, mode='nearest')
            rvD = rvD * gain; Rd = R.from_rotvec(rvD).as_matrix()
            # keep the moved object-centre translation (smoothed), rebuild D around it
            cO = np.einsum('ij,tj->ti', Tch[:3, :3], (Told[:, :3, 3] if Tn <= n else np.concatenate([Told[:, :3, 3], np.repeat(Told[-1:, :3, 3], Tn - n, 0)]))[:Tn]) + Tch[:3, 3]
            if sig > 0 and not a.rot_only: cD = gaussian_filter1d(cD, sig, axis=0, mode='nearest')
            D = np.tile(np.eye(4), (Tn, 1, 1)); D[:, :3, :3] = Rd; D[:, :3, 3] = cD - np.einsum('tij,tj->ti', Rd, cO)
            Tnew = Tnew.copy(); Tnew[:min(n, Tn), :3, 3] = cD[:min(n, Tn)]
        rv = R.from_matrix(D[:, :3, :3]).as_rotvec(); Dr = R.from_rotvec(rv * ramp[:, None]).as_matrix()
        # translation: interpolate the moved wrist position between "carried only" and "fully moved"
        moved = np.einsum('tij,tj->ti', D[:, :3, :3], Wt[:, :3, 3]) + D[:, :3, 3]
        rot_moved = np.einsum('tij,tjk->tik', Dr, Wt[:, :3, :3])
        # position follows the same partial rotation about the object target centre plus the ramped residual translation
        c = np.einsum('ij,tj->ti', Tch[:3, :3], Told[:, :3, 3] if Tn <= n else np.concatenate([Told[:, :3, 3], np.repeat(Told[-1:, :3, 3], Tn - n, 0)])) + Tch[:3, 3]
        cn = Tnew[:, :3, 3] if Tn <= n else np.concatenate([Tnew[:, :3, 3], np.repeat(Tnew[-1:, :3, 3], Tn - n, 0)])
        p_partial = np.einsum('tij,tj->ti', Dr, Wt[:, :3, 3] - c[:Tn]) + c[:Tn] + ramp[:, None] * (cn[:Tn] - c[:Tn])
        assert np.allclose(p_partial[ramp == 1], moved[ramp == 1], atol=1e-9)
        Q[:, cols[:3]] = p_partial; Q[:, cols[3:]] = continuous_euler_xyz(R.from_matrix(rot_moved), Qnew[0, cols[3:]])
        report[side] = dict(object=obj, delta_cm_mean=float(np.linalg.norm(p_partial - Wt[:, :3, 3], axis=1).mean() * 100), delta_cm_max=float(np.linalg.norm(p_partial - Wt[:, :3, 3], axis=1).max() * 100),
                            delta_deg_mean=float(np.degrees(np.linalg.norm(rv, axis=1)).mean()), delta_deg_max=float(np.degrees(np.linalg.norm(rv, axis=1)).max()))
    obj_cols = [i for i, nm in enumerate(names) if not nm.startswith('robot_')]; nn = min(Tn, len(Qnew)); Q[:nn, obj_cols] = Qnew[:nn, obj_cols]
    lim = m.jnt_range[m.actuator_trnid[:, 0]]; qa = m.jnt_qposadr[m.actuator_trnid[:, 0]]; Q[:, qa] = np.clip(Q[:, qa], lim[:, 0], lim[:, 1]); Q[0] = Qnew[0]
    refp = TR.save_reference(out / 'reference.npz', m, Q, list(tref['labels']), fps, tref['segment'], tref['segments'], source=f'object-relative retarget of {teacher} into {new}', arms_coupled=False, floating_runtime=True)
    ex = execute_reference(str(scene), refp, out / 'execution'); man = json.loads((new / 'input_manifest.json').read_text())
    met = evaluate_files(man['source'], ex['trajectory'], new / 'input_manifest.json', ex['scene'], new / 'metric_geometry.json')
    res = dict(episode='episode_000008', input_dir=str(new), reference=str(refp), execution=ex, metrics=met, learned_policy=False,
               retarget=dict(teacher=str(teacher), teacher_result_sha256=sha(teacher / 'result.json'), old_epdir=str(old), new_epdir=str(new), setup_change=Tch.tolist(), assign=a.assign, ramp_s=a.ramp_s, hands=report, scene=str(scene), scene_sha256=sha(scene)),
               spec=dict(episode='episode_000008', scene=str(scene), reference=str(refp), source=man['source'], manifest=str(new / 'input_manifest.json'), geometry=str(new / 'metric_geometry.json'), original_reference=str(new / 'tracking_reference.npz')))
    (out / 'result.json').write_text(json.dumps(res, indent=1, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)) + '\n')
    o = {k: (round(v['mean_position_error_cm'][0], 2), round(v['mean_orientation_error_degrees'][0], 1)) for k, v in met['official_style_reference_proxy']['objects'].items()}
    print(json.dumps(dict(out=str(out), hands=report, mp_sr=met['reference_proxy_mp_sr'], objects=o, completed=ex['execution_completed'], physics_verified=ex['physics_verified'])), flush=True)
if __name__ == '__main__': main()
