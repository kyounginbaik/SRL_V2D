"""Stream D (v22): port a teacher (dir with result.json + reference.npz) into the target_final_v5 world of its episode.
Generalised from streamT_C356/code/port.py:
- scene: the teacher's frozen scene_physics.xml with the object bodies / home keyframe moved to the new frame-0 poses; with
  --home-robot the wrist joints of those sides in the home keyframe also move to the ported reference frame 0 (needed when a hand
  already holds an object at frame 0, ep20 left hand + cup).
- reference: robot columns of the teacher; each listed wrist is retargeted object-relative,
      wrist_new(t) = D(t) @ wrist_old(t),  D(t) = T_new(t) @ inv(T_old(t))     (Gaussian-smoothed, --sigma frames)
  old = the teacher's ACHIEVED object trajectory (--old ach, default: the hand keeps the relation it physically had) or the old
  episode dir's tracking reference (--old ref); new = target reference of the new world. Object columns = new targets.
  Optional --pos-gain / --rot-frac per side scale D (fraction of the displacement applied).
- executes with the unchanged floating runtime and evaluates against the new episode dir; result.json in the usual tool schema.
usage: port_d.py EP TEACHER_DIR NEW_EPISODE_DIR OUT_DIR --pair left=OBJ,right=OBJ [--sides left,right] [--old ach|ref --old-epd DIR]
       [--sigma 1.5] [--rot-frac 1,1] [--home-robot left] [--t0 S --t1 S (apply D only in [t0,t1] s, with --ramp s blends)]"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import sys, json, re, argparse
from pathlib import Path
import numpy as np, mujoco
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R
LEG = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017'); sys.path[:0] = [str(LEG / 'code'), '/data1/SRL_V2D/third_party/DexAgent']
from tool_library.common import hand as H, tracking as TR
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files
from wrist_reference import continuous_euler_xyz
def obj_T(q, c):
    T = np.tile(np.eye(4), (len(q), 1, 1)); T[:, :3, 3] = q[:, c:c + 3]; T[:, :3, :3] = R.from_quat(q[:, c + 3:c + 7], scalar_first=True).as_matrix(); return T
def smooth_D(D, sig):
    if sig <= 0: return D
    t = gaussian_filter1d(D[:, :3, 3], sig, axis=0, mode='nearest'); r0 = R.from_matrix(D[0, :3, :3])
    rv = gaussian_filter1d((r0.inv() * R.from_matrix(D[:, :3, :3])).as_rotvec(), sig, axis=0, mode='nearest')
    out = D.copy(); out[:, :3, 3] = t; out[:, :3, :3] = (r0 * R.from_rotvec(rv)).as_matrix(); return out
def scale_D(D, w):
    """D^w (per frame weight w in [0,1]) about the identity"""
    rv = R.from_matrix(D[:, :3, :3]).as_rotvec() * w[:, None]; out = D.copy(); out[:, :3, :3] = R.from_rotvec(rv).as_matrix(); out[:, :3, 3] = D[:, :3, 3] * w[:, None]; return out
def _sym(ep, objs):
    sys.path.insert(0, str(Path(__file__).resolve().parent)); import nlib as NL
    X = NL.registration(ep); out = {}
    for o in objs:
        ax = NL.K.symmetry_axis(o, NL.ref_mesh(ep, o)[0])[0]; ab = X[o][:3, :3].T @ ax; out[o] = (ab / np.linalg.norm(ab), NL.CA[(ep, o)])
    return out
SYM = {}
def main():
    p = argparse.ArgumentParser(); p.add_argument('ep', type=int); p.add_argument('teacher'); p.add_argument('epd'); p.add_argument('out')
    p.add_argument('--pair', required=True); p.add_argument('--sides', default='left,right'); p.add_argument('--old', default='ach'); p.add_argument('--old-epd')
    p.add_argument('--sigma', type=float, default=1.5); p.add_argument('--rot-frac', default='1,1'); p.add_argument('--home-robot', default=''); p.add_argument('--freeze', default='', help='side=seconds[,...]: D held constant after that time (hand releases relative to where the object was put down)')
    p.add_argument('--t0', type=float, default=-1); p.add_argument('--t1', type=float, default=1e9); p.add_argument('--ramp', type=float, default=0.5)
    p.add_argument('--sym', default='', help='axisymmetric objects (comma list): swing-only D about the scorer axis point (port_g.py mode)'); p.add_argument('--scene', help='use this (already new-world) scene instead of deriving one from the teacher'); a = p.parse_args()
    PAIR = dict(x.split('=') for x in a.pair.split(','))
    global SYM; SYM = _sym(a.ep, list(filter(None, a.sym.split(',')))) if a.sym else {}
    teacher, out, epd = Path(a.teacher).resolve(), Path(a.out).resolve(), Path(a.epd).resolve(); out.mkdir(parents=True, exist_ok=False)
    tr = json.loads((teacher / 'result.json').read_text()); refp = Path(tr['reference']) if 'reference' in tr and Path(tr['reference']).exists() else teacher / 'reference.npz'
    m = mujoco.MjModel.from_xml_path(tr['execution']['scene']); ref = TR.load_reference(refp, m); Q = ref['qpos'].copy(); n = len(Q)
    new = TR.load_reference(epd / 'tracking_reference.npz', m)['qpos']; assert len(new) == n
    if a.old == 'ref': oldq = TR.load_reference(Path(a.old_epd) / 'tracking_reference.npz', m)['qpos']
    else:
        ot = np.load(tr['execution']['trajectory']); oldq = ot['qpos'][:n]; assert len(oldq) == n, (len(oldq), n)
    t = np.arange(n) / ref['fps']
    w = np.clip(np.minimum((t - a.t0) / max(a.ramp, 1e-6) + 1, (a.t1 - t) / max(a.ramp, 1e-6) + 1), 0, 1) if (a.t0 >= 0 or a.t1 < 1e8) else np.ones(n)
    info = {}
    for side in filter(None, a.sides.split(',')):
        o = PAIR[side]; c = int(m.joint(o + '_free').qposadr[0])
        if o in SYM:   # axisymmetric: spin about the symmetry axis is meaningless (scorer order 360) -> swing-only D about the axis point
            ab, ca = SYM[o]; To, Tn = obj_T(oldq, c), obj_T(new, c)
            ao = To[:, :3, :3] @ ab; an = Tn[:, :3, :3] @ ab; cr = np.cross(ao, an); sn = np.linalg.norm(cr, axis=1, keepdims=True)
            rv = cr / np.maximum(sn, 1e-12) * np.arctan2(sn[:, 0], (ao * an).sum(1))[:, None]; Rd = R.from_rotvec(rv).as_matrix()
            po = np.einsum('nij,j->ni', To[:, :3, :3], ca) + To[:, :3, 3]; pn = np.einsum('nij,j->ni', Tn[:, :3, :3], ca) + Tn[:, :3, 3]
            D = np.tile(np.eye(4), (n, 1, 1)); D[:, :3, :3] = Rd; D[:, :3, 3] = pn - np.einsum('nij,nj->ni', Rd, po); D = smooth_D(D, a.sigma)
        else: D = smooth_D(obj_T(new, c) @ np.linalg.inv(obj_T(oldq, c)), a.sigma)
        fr = float(a.rot_frac.split(',')[0 if side == 'left' else 1])
        if fr != 1.:   # object position still maps exactly, only part of the rotation is applied
            co = gaussian_filter1d(oldq[:, c:c + 3], a.sigma, axis=0, mode='nearest'); cn = np.einsum('nij,nj->ni', D[:, :3, :3], co) + D[:, :3, 3]
            Rf = R.from_rotvec(fr * R.from_matrix(D[:, :3, :3]).as_rotvec()).as_matrix(); D = D.copy(); D[:, :3, :3] = Rf; D[:, :3, 3] = cn - np.einsum('nij,nj->ni', Rf, co)
        fz = dict(x.split('=') for x in filter(None, a.freeze.split(',')))
        if side in fz:
            kf = int(round(float(fz[side]) * ref['fps'])); D[kf:] = D[kf]
        D = scale_D(D, w)
        rig = H.HandRig.build(m, f'robot_{side}_', side); wc = np.array([rig.wrist_qpos[k] for k in ('x', 'y', 'z', 'rx', 'ry', 'rz')])
        Wp = Q[:, wc[:3]].copy(); Rw = R.from_euler('XYZ', Q[:, wc[3:]])
        Q[:, wc[:3]] = np.einsum('nij,nj->ni', D[:, :3, :3], Wp) + D[:, :3, 3]
        Q[:, wc[3:]] = continuous_euler_xyz(R.from_matrix(D[:, :3, :3] @ Rw.as_matrix()), Q[0, wc[3:]])
        info[side] = dict(object=o, wrist_shift_mean_cm=float(np.linalg.norm(Q[:, wc[:3]] - Wp, axis=1).mean() * 100), wrist_shift_f0_cm=float(np.linalg.norm(Q[0, wc[:3]] - Wp[0]) * 100))
    for o in PAIR.values():
        c = int(m.joint(o + '_free').qposadr[0]); Q[:, c:c + 7] = new[:, c:c + 7]
    lim = m.jnt_range[m.actuator_trnid[:, 0]]; qa = m.jnt_qposadr[m.actuator_trnid[:, 0]]; Q[:, qa] = np.clip(Q[:, qa], lim[:, 0], lim[:, 1])
    scene = Path(a.scene) if a.scene else out / 'scene_target.xml'
    if not a.scene:
        s = Path(tr['execution']['scene']).read_text()
        for o in PAIR.values():
            c = int(m.joint(o + '_free').qposadr[0]); pp = new[0, c:c + 3]; qq = new[0, c + 3:c + 7]
            s, k = re.subn(rf'(<body name="{o}" )pos="[^"]*" quat="[^"]*"', lambda mm: mm.group(1) + f'pos="{pp[0]:.9g} {pp[1]:.9g} {pp[2]:.9g}" quat="{qq[0]:.9g} {qq[1]:.9g} {qq[2]:.9g} {qq[3]:.9g}"', s); assert k == 1, (o, k)
        kq = re.search(r'<key name="home" qpos="([^"]*)"', s); vals = np.array(kq.group(1).split(), float)
        for o in PAIR.values():
            c = int(m.joint(o + '_free').qposadr[0]); vals[c:c + 7] = new[0, c:c + 7]
        for side in filter(None, a.home_robot.split(',')):   # home wrist moved rigidly with its object's frame-0 displacement
            o = PAIR[side]; c = int(m.joint(o + '_free').qposadr[0]); hv = vals.copy(); hv[c:c + 7] = (oldq[0] if a.old == 'ach' else vals)[c:c + 7]
            D0 = obj_T(new[:1], c)[0] @ np.linalg.inv(obj_T(hv[None], c)[0])
            rig = H.HandRig.build(m, f'robot_{side}_', side); wc = np.array([rig.wrist_qpos[k] for k in ('x', 'y', 'z', 'rx', 'ry', 'rz')])
            vals[wc[:3]] = D0[:3, :3] @ vals[wc[:3]] + D0[:3, 3]
            vals[wc[3:]] = R.from_matrix(D0[:3, :3] @ R.from_euler('XYZ', vals[wc[3:]]).as_matrix()).as_euler('XYZ')
            info.setdefault('home_robot', {})[side] = dict(D0_cm=float(np.linalg.norm(D0[:3, 3]) * 100), D0_deg=float(np.degrees(R.from_matrix(D0[:3, :3]).magnitude())))
        s = s[:kq.start(1)] + ' '.join(f'{v:.9g}' for v in vals) + s[kq.end(1):]; scene.write_text(s)
    rp = TR.save_reference(out / 'reference.npz', m, Q, ref['labels'], ref['fps'], ref['segment'], ref['segments'], source=f'stream D port of {teacher} (old={a.old}, sigma={a.sigma})', arms_coupled=False, floating_runtime=True)
    ex = execute_reference(str(scene), rp, out / 'execution'); man = json.loads((epd / 'input_manifest.json').read_text())
    met = evaluate_files(man['source'], ex['trajectory'], epd / 'input_manifest.json', ex['scene'], epd / 'metric_geometry.json')
    res = dict(episode=f'episode_{a.ep:06d}', input_dir=str(epd), reference=str(rp), execution=ex, metrics=met, parent=str(teacher), teacher_metrics=tr.get('metrics'),
               port=dict(vars(a), info=info, scene=str(scene)), learned_policy=False)
    (out / 'result.json').write_text(json.dumps(res, indent=1, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)) + '\n')
    fmt = {k: (round(v['mean_position_error_cm'][0], 2), round(v['mean_orientation_error_degrees'][0], 1), round(v['max_position_error_m'][0] * 100, 1)) for k, v in met['official_style_reference_proxy']['objects'].items()}
    print(json.dumps(dict(out=str(out), mp_sr=met['reference_proxy_mp_sr'], objects=fmt, verified=ex['physics_verified'], info=info)), flush=True)
if __name__ == '__main__': main()
