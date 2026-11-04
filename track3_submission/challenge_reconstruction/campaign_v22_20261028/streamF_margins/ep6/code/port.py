"""Stream T: port a v21 teacher (dir with result.json + reference.npz) into the compose-referenced world of its episode.
- scene: the teacher's frozen scene_physics.xml with the object bodies / home keyframe moved to the new frame-0 poses (build_world.fix_scene logic)
- reference: robot columns of the teacher; with --follow each wrist is retargeted object-relative,
      wrist_new(t) = D_o(t) @ wrist_old(t),  D_o(t) = T_new_o(t) @ inv(T_old_o(t))   (left wrist <- white_pot, right wrist <- wooden_spoon;
      old = the v21 world's reference, new = compose reference; D smoothed with a Gaussian of --sigma frames); object columns = new targets
- executes with the unchanged floating runtime and scores against the compose episode dir. Output result.json in the usual tool schema.
usage: port.py EP TEACHER_DIR OLD_EPISODE_DIR OUT_DIR [--follow] [--sigma 1.5] [--sides left,right]"""
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
ST = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamF_margins')
PAIR = {'left': 'white_pot', 'right': 'wooden_spoon'}
def world_dir(ep): return ST / f'ep{ep}/world/episode_compose/episode_{ep:06d}'
def obj_T(q, c):
    T = np.tile(np.eye(4), (len(q), 1, 1)); T[:, :3, 3] = q[:, c:c + 3]; T[:, :3, :3] = R.from_quat(q[:, c + 3:c + 7], scalar_first=True).as_matrix(); return T
def fix_scene(src_xml, dst_xml, q0, m):
    s = Path(src_xml).read_text()
    for o in PAIR.values():
        c = int(m.joint(o + '_free').qposadr[0]); p = q0[c:c + 3]; qq = q0[c + 3:c + 7]
        s, k = re.subn(rf'(<body name="{o}" )pos="[^"]*" quat="[^"]*"', lambda mm: mm.group(1) + f'pos="{p[0]:.9g} {p[1]:.9g} {p[2]:.9g}" quat="{qq[0]:.9g} {qq[1]:.9g} {qq[2]:.9g} {qq[3]:.9g}"', s); assert k == 1, (o, k)
    kq = re.search(r'<key name="home" qpos="([^"]*)"', s); vals = np.array(kq.group(1).split(), float)
    for o in PAIR.values():
        c = int(m.joint(o + '_free').qposadr[0]); vals[c:c + 7] = q0[c:c + 7]
    s = s[:kq.start(1)] + ' '.join(f'{v:.9g}' for v in vals) + s[kq.end(1):]; Path(dst_xml).write_text(s)
def unwrap_rv(rv):
    """stream A fix: make a rotation-vector sequence continuous through |angle| = pi (the v22 ep3 target spoon rolls a full turn;
    the phase-6 smoothing averaged across the rotvec wrap)."""
    out = rv.copy()
    for k in range(1, len(rv)):
        v = rv[k]; a = np.linalg.norm(v)
        if a < 1e-9: continue
        u = v / a; cands = [v + 2 * np.pi * j * u for j in (-2, -1, 0, 1, 2)]
        out[k] = min(cands, key=lambda c: np.linalg.norm(c - out[k - 1]))
    return out
def smooth_D(D, sig):
    if sig <= 0: return D
    t = gaussian_filter1d(D[:, :3, 3], sig, axis=0, mode='nearest'); r0 = R.from_matrix(D[0, :3, :3])
    rv = gaussian_filter1d(unwrap_rv((r0.inv() * R.from_matrix(D[:, :3, :3])).as_rotvec()), sig, axis=0, mode='nearest')
    out = D.copy(); out[:, :3, 3] = t; out[:, :3, :3] = (r0 * R.from_rotvec(rv)).as_matrix(); return out
def contact_window(m, traj, o, side):
    d = mujoco.MjData(m); hit = []
    for k in range(len(traj['qpos'])):
        d.qpos[:] = traj['qpos'][k]; mujoco.mj_forward(m, d); h = False
        for cc in d.contact[:d.ncon]:
            b1, b2 = m.body(int(m.geom_bodyid[cc.geom1])).name, m.body(int(m.geom_bodyid[cc.geom2])).name
            if cc.dist < 0.002 and ((b1 == o and b2.startswith('robot_' + side)) or (b2 == o and b1.startswith('robot_' + side))): h = True; break
        hit.append(h)
    hit = np.flatnonzero(hit); return int(hit[0]), int(hit[-1])
def smooth_T(T, sig):
    """smooth a pose track: positions gaussian, rotations = gaussian on sign-continuous quaternions (renormalised); robust through full turns."""
    if sig <= 0: return T.copy()
    q = R.from_matrix(T[:, :3, :3]).as_quat()
    for k in range(1, len(q)):
        if q[k] @ q[k - 1] < 0: q[k] = -q[k]
    q = gaussian_filter1d(q, sig, axis=0, mode='nearest'); q /= np.linalg.norm(q, axis=1, keepdims=True)
    out = T.copy(); out[:, :3, :3] = R.from_quat(q).as_matrix(); out[:, :3, 3] = gaussian_filter1d(T[:, :3, 3], sig, axis=0, mode='nearest'); return out
def obj_D(m, tr, a, o, side, new, c, n):
    """stream A mode 'obj': interpolate in OBJECT pose space. Target object pose T_tgt(t):
       t < g0 (first teacher contact): settled new rest pose Ts (the hand approaches it exactly as in the teacher);
       t >= g0: blend from the old achieved motion rigidly shifted onto the rest pose (Ts @ inv(To[g0]) @ To(t)) to the smoothed
       new reference, weight w = alpha * smoothstep((t - g0 - obj_start) / blend); rotation blended along the unwrapped
       relative rotation vector (so a full-turn roll is followed); D(t) = T_tgt(t) @ inv(To(t)) carries the wrist with the object.
       After the last contact g1 D is frozen."""
    old_tr = np.load(tr['execution']['trajectory']); i = [str(x) for x in old_tr['object_names']].index(o)
    To = np.tile(np.eye(4), (n, 1, 1)); To[:, :3, 3] = old_tr['object_positions'][:n, i]; To[:, :3, :3] = R.from_quat(old_tr['object_quaternions_wxyz'][:n, i][:, [1, 2, 3, 0]]).as_matrix()
    st = np.load(json.loads((Path(a.settled) / 'result.json').read_text())['execution']['trajectory']); j = [str(x) for x in st['object_names']].index(o)
    Ts = np.eye(4); Ts[:3, 3] = np.median(st['object_positions'][10:26, j], 0); Ts[:3, :3] = R.from_quat(st['object_quaternions_wxyz'][18, j][[1, 2, 3, 0]]).as_matrix()
    g0, g1 = contact_window(m, old_tr, o, side); Tn = smooth_T(obj_T(new, c), a.sigma)
    rr = float(a.obj_restrot)   # fraction of the settled-vs-old rest rotation applied (about the object centre); 0 = translate only
    if rr != 1.:
        Ro = To[g0, :3, :3]; rvr = R.from_matrix(Ts[:3, :3] @ Ro.T).as_rotvec(); Ts = Ts.copy(); Ts[:3, :3] = R.from_rotvec(rr * rvr).as_matrix() @ Ro
    if a.obj_dz: Ts = Ts.copy(); Ts[2, 3] += a.obj_dz
    al = float(a.alpha.split(',')[0 if side == 'left' else 1]); st0 = float(a.obj_start.split(',')[0 if side == 'left' else 1])
    x = np.clip((np.arange(n) - g0 - st0 * 20) / max(1., a.blend * 20), 0, 1); w = x * x * (3 - 2 * x) * al
    S0 = Ts @ np.linalg.inv(To[min(g0, n - 1)]); Tf = np.einsum('ij,njk->nik', S0, To)
    rv = unwrap_rv(R.from_matrix(np.einsum('nji,njk->nik', Tf[:, :3, :3], Tn[:, :3, :3])).as_rotvec())
    D = np.tile(np.eye(4), (n, 1, 1))
    for k in range(n):
        if k < g0: Tt = Ts
        else:
            Tt = np.eye(4); Tt[:3, :3] = Tf[k, :3, :3] @ R.from_rotvec(w[k] * rv[k]).as_matrix(); Tt[:3, 3] = (1 - w[k]) * Tf[k, :3, 3] + w[k] * Tn[k, :3, 3]
        D[k] = Tt @ np.linalg.inv(To[k])
    D[g1 + 1:] = D[g1]
    print(json.dumps(dict(side=side, object=o, contact=[g0, g1], mode='obj', alpha=al, start=st0)), flush=True)
    return D
def ach_D(m, tr, a, o, side, new, c, n):
    """D(t) = A(t) @ inv(T_old_achieved(t)): A = physically settled new rest pose before the grasp, the compose reference while held
    (blended in over --blend s after first contact), frozen after the last contact; smoothed with --sigma frames."""
    old_tr = np.load(tr['execution']['trajectory']); i = [str(x) for x in old_tr['object_names']].index(o)
    To = np.tile(np.eye(4), (n, 1, 1)); To[:, :3, 3] = old_tr['object_positions'][:n, i]; To[:, :3, :3] = R.from_quat(old_tr['object_quaternions_wxyz'][:n, i][:, [1, 2, 3, 0]]).as_matrix()
    st = np.load(json.loads((Path(a.settled) / 'result.json').read_text())['execution']['trajectory']); j = [str(x) for x in st['object_names']].index(o)
    Ts = np.eye(4); Ts[:3, 3] = np.median(st['object_positions'][10:26, j], 0); Ts[:3, :3] = R.from_quat(st['object_quaternions_wxyz'][18, j][[1, 2, 3, 0]]).as_matrix()
    g0, g1 = contact_window(m, old_tr, o, side); Tn = obj_T(new, c)
    Dref = smooth_D(Tn @ np.linalg.inv(To), a.sigma); Drest = Ts @ np.linalg.inv(To)
    rs = a.rest.split(','); rest = rs[0] if len(rs) == 1 else rs[0 if side == 'left' else 1]
    if rest == 'identity': Drest = np.tile(np.eye(4), (n, 1, 1))
    if rest == 'ref':   # pre-grasp: reference-to-reference displacement (as mode ref)
        old_epd = Path(a.old_epd); oq = TR.load_reference(old_epd / 'tracking_reference.npz', m)['qpos']; Drest = smooth_D(Tn @ np.linalg.inv(obj_T(oq, c)), 1.5)
    w = np.clip((np.arange(n) - g0) / max(1., a.blend * 20), 0, 1); w = w * w * (3 - 2 * w) * float(a.alpha.split(',')[0 if side == 'left' else 1])
    D = Dref.copy()
    for k in range(n):
        if k > g1: D[k] = D[g1]; continue
        Dr = Drest[k]; Dn = Dref[k]; rv = R.from_matrix(Dn[:3, :3] @ Dr[:3, :3].T).as_rotvec()
        D[k, :3, :3] = R.from_rotvec(w[k] * rv).as_matrix() @ Dr[:3, :3]; D[k, :3, 3] = (1 - w[k]) * Dr[:3, 3] + w[k] * Dn[:3, 3]
    fr = float(a.rot_frac.split(',')[0 if side == 'left' else 1])
    if fr != 1.:   # keep where the old achieved object position maps, apply only part of the rotation
        co = To[:, :3, 3]; cn = np.einsum('nij,nj->ni', D[:, :3, :3], co) + D[:, :3, 3]
        Rf = R.from_rotvec(fr * R.from_matrix(D[:, :3, :3]).as_rotvec()).as_matrix(); D[:, :3, :3] = Rf; D[:, :3, 3] = cn - np.einsum('nij,nj->ni', Rf, co)
    print(json.dumps(dict(side=side, object=o, contact=[g0, g1], rot_frac=fr)), flush=True)
    return D
def main():
    p = argparse.ArgumentParser(); p.add_argument('ep', type=int); p.add_argument('teacher'); p.add_argument('old_epd'); p.add_argument('out')
    p.add_argument('--follow', action='store_true'); p.add_argument('--sigma', type=float, default=1.5); p.add_argument('--sides', default='left,right'); p.add_argument('--rot-frac', default='1,1', help='rotation fraction of D for left,right')
    p.add_argument('--mode', default='ref', help='ref|ach or a left,right pair'); p.add_argument('--settled', help='run dir in the compose world whose frames 10-25 give the physically settled frame-0 object poses (mode ach)'); p.add_argument('--blend', type=float, default=0.5); p.add_argument('--obj-start', default='0.5,0.5', help='mode obj: blend start (s after first contact), left,right'); p.add_argument('--obj-restrot', type=float, default=1.); p.add_argument('--obj-dz', type=float, default=0.); p.add_argument('--alpha', default='1,1', help='left,right: fraction of the compose-following during the hold (0 = old motion shifted by the rest offset)'); p.add_argument('--rest', default='settled', help='settled|ref, or a left,right pair')
    p.add_argument('--home-old', default='', help='objects whose scene home stays at the OLD world frame-0 reference pose (e.g. wooden_spoon)')
    p.add_argument('--scene', help='use this scene (already compose-world) instead of deriving one from the teacher'); a = p.parse_args()
    teacher, out = Path(a.teacher).resolve(), Path(a.out).resolve(); epd = world_dir(a.ep); out.mkdir(parents=True, exist_ok=False)
    tr = json.loads((teacher / 'result.json').read_text()); refp = Path(tr['reference']) if 'reference' in tr and Path(tr['reference']).exists() else teacher / 'reference.npz'
    m = mujoco.MjModel.from_xml_path(tr['execution']['scene']); ref = TR.load_reference(refp, m); Q = ref['qpos'].copy(); n = len(Q)
    new = TR.load_reference(epd / 'tracking_reference.npz', m)['qpos']; old = TR.load_reference(Path(a.old_epd) / 'tracking_reference.npz', m)['qpos']
    assert len(new) == n == len(old)
    scene = Path(a.scene) if a.scene else out / 'scene_compose.xml'
    if not a.scene:
        q0 = new[0].copy()
        for o in filter(None, a.home_old.split(',')):
            c = int(m.joint(o + '_free').qposadr[0]); q0[c:c + 7] = old[0, c:c + 7]
        fix_scene(tr['execution']['scene'], scene, q0, m)
    info = {}
    if a.follow:
        for side in a.sides.split(','):
            o = PAIR[side]; c = int(m.joint(o + '_free').qposadr[0])
            md = a.mode.split(','); md = md[0] if len(md) == 1 else md[0 if side == 'left' else 1]
            if md == 'ach':
                D = ach_D(m, tr, a, o, side, new, c, n)
            elif md == 'obj':
                D = obj_D(m, tr, a, o, side, new, c, n)
            else:
                D = smooth_D(obj_T(new, c) @ np.linalg.inv(obj_T(old, c)), a.sigma)
            fr = 1. if md in ('ach', 'obj') else float(a.rot_frac.split(',')[0 if side == 'left' else 1])
            if fr != 1.:   # D = Trans(c_new) R^fr Trans(-c_old): the object position still maps exactly, only part of the rotation is applied
                Rf = R.from_rotvec(fr * R.from_matrix(D[:, :3, :3]).as_rotvec()).as_matrix(); cn = gaussian_filter1d(new[:, c:c + 3], a.sigma, axis=0, mode='nearest') if a.sigma > 0 else new[:, c:c + 3]
                co = gaussian_filter1d(old[:, c:c + 3], a.sigma, axis=0, mode='nearest') if a.sigma > 0 else old[:, c:c + 3]
                D = D.copy(); D[:, :3, :3] = Rf; D[:, :3, 3] = cn - np.einsum('nij,nj->ni', Rf, co)
            rig = H.HandRig.build(m, f'robot_{side}_', side); wc = np.array([rig.wrist_qpos[k] for k in ('x', 'y', 'z', 'rx', 'ry', 'rz')])
            Wp = Q[:, wc[:3]]; Rw = R.from_euler('XYZ', Q[:, wc[3:]])
            Q[:, wc[:3]] = np.einsum('nij,nj->ni', D[:, :3, :3], Wp) + D[:, :3, 3]
            Q[:, wc[3:]] = continuous_euler_xyz(R.from_matrix(D[:, :3, :3] @ Rw.as_matrix()), Q[0, wc[3:]])
            info[side] = dict(object=o, wrist_shift_mean_cm=float(np.linalg.norm(Q[:, wc[:3]] - Wp, axis=1).mean() * 100))
    for o in PAIR.values():
        c = int(m.joint(o + '_free').qposadr[0]); Q[:, c:c + 7] = new[:, c:c + 7]
    lim = m.jnt_range[m.actuator_trnid[:, 0]]; qa = m.jnt_qposadr[m.actuator_trnid[:, 0]]; Q[:, qa] = np.clip(Q[:, qa], lim[:, 0], lim[:, 1])
    rp = TR.save_reference(out / 'reference.npz', m, Q, ref['labels'], ref['fps'], ref['segment'], ref['segments'], source=f'stream T port of {teacher} (follow={a.follow}, sigma={a.sigma})', arms_coupled=False, floating_runtime=True)
    ex = execute_reference(str(scene), rp, out / 'execution'); man = json.loads((epd / 'input_manifest.json').read_text())
    met = evaluate_files(man['source'], ex['trajectory'], epd / 'input_manifest.json', ex['scene'], epd / 'metric_geometry.json')
    res = dict(episode=f'episode_{a.ep:06d}', input_dir=str(epd), reference=str(rp), execution=ex, metrics=met, parent=str(teacher), teacher_metrics=tr.get('metrics'),
               port=dict(home_old=a.home_old, follow=a.follow, rot_frac=a.rot_frac, mode=a.mode, settled=a.settled, blend=a.blend, obj_start=a.obj_start, obj_restrot=a.obj_restrot, obj_dz=a.obj_dz, rest=a.rest, alpha=a.alpha, sigma=a.sigma, sides=a.sides, info=info, scene=str(scene), old_episode_dir=str(a.old_epd)), learned_policy=False)
    (out / 'result.json').write_text(json.dumps(res, indent=1, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)) + '\n')
    fmt = {k: (round(v['mean_position_error_cm'][0], 2), round(v['mean_orientation_error_degrees'][0], 1), round(v['max_position_error_m'][0] * 100, 1)) for k, v in met['official_style_reference_proxy']['objects'].items()}
    print(json.dumps(dict(out=str(out), mp_sr=met['reference_proxy_mp_sr'], objects=fmt, verified=ex['physics_verified'], info=info)), flush=True)
if __name__ == '__main__': main()
