"""J8 ep19 copy of streamF port19.py (unchanged logic; --episode-dir/--scene always passed). Object-relative port of a teacher reference to the compose targets: each carrying wrist is moved rigidly with its object,
  W'[t] = D[t] @ W[t],  D[t] = smooth( new_ref[t] @ inv(achieved_old[t]) )
(achieved_old = the teacher's own physical object trajectory, so the realised hand-object relation is kept; before contact both
are static and D is the constant frame-0 delta). Fingers unchanged. Executed in the compose-home scene, evaluated against the
compose-corrected episode dir; output dir is a teacher dir (result.json + reference.npz + execution/) for ilc_carry_generic.py.
usage: port_relative.py TEACHER OUT --pair right:mini_sweeper [--pair left:mini_dust_pan] [--sigma 3] [--gain 1.0] [--t-from S]"""
import argparse, json, shutil, sys
from pathlib import Path
import numpy as np, mujoco
from scipy.ndimage import gaussian_filter1d
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamF_margins/ep19/code')
from common19F import *
ROOT = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017'); sys.path[:0] = [str(ROOT / 'code'), '/data1/SRL_V2D/third_party/DexAgent']
from tool_library.common import hand as H, tracking as TR
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files
from wrist_reference import continuous_euler_xyz
GTW = np.array(json.loads(Path('/data1/SRL_V2D/challenge_reconstruction/recon_v15_20261022/episode_000019/change.json').read_text())['G'])
ap = argparse.ArgumentParser(); ap.add_argument('teacher'); ap.add_argument('out'); ap.add_argument('--pair', action='append', required=True)
ap.add_argument('--sigma', type=float, default=3.); ap.add_argument('--gain', type=float, default=1.); ap.add_argument('--t-from', type=float, default=0.); ap.add_argument('--pre', action='append', default=[], help='obj:TRAJ:FRAME:T0:BLEND[:T1:BLEND1:G]'); ap.add_argument('--pre-rot-scale', action='append', default=[], help='obj:F scale of the rotation part of the pre->compose blend (0 = translation-only follow)'); ap.add_argument('--lift', action='append', default=[], help='side:DZ_M:T0:T1 world +z wrist offset (0.3 s ramps), table clearance'); ap.add_argument('--squeeze', action='append', default=[], help='side:RAD:T0:T1 add RAD to the flexion joints (MCP_FE/PIP/DIP/thumb CMC_FE, MCP_FE, IP) of that hand between T0 and T1 (0.3 s ramps)')
ap.add_argument('--episode-dir', default=str(V / 'world_tw/episode_target/episode_000019')); ap.add_argument('--scene', default=str(V / 'world_tw/scene/scene_physics_compose_home.xml'))
ap.add_argument('--twin', action='append', default=[], help='object whose OLD achieved trajectory is relabelled by the half-turn twin G (the teacher ran on the other face; the hand keeps its world grasp)')
a = ap.parse_args(); PRE = {x.split(':')[0]: (x.split(':')[1], int(x.split(':')[2]), *map(float, x.split(':')[3:])) for x in a.pre}
sm = lambda x: np.clip(x, 0, 1) ** 3 * (10 - 15 * np.clip(x, 0, 1) + 6 * np.clip(x, 0, 1) ** 2)
src, out, epd = Path(a.teacher), Path(a.out), Path(a.episode_dir)
prev = json.loads((src / 'result.json').read_text()); m = mujoco.MjModel.from_xml_path(a.scene)
ref = TR.load_reference(src / 'reference.npz', m); Q = ref['qpos'].copy(); fps = ref['fps']; T = len(Q)
new = TR.load_reference(epd / 'tracking_reference.npz', m)['qpos']; tr = np.load(prev['execution']['trajectory']); on = list(tr['object_names']); n = min(T, len(tr['object_positions']))
rec = {}
for pr in a.pair:
    side, obj = pr.split(':'); rig = H.HandRig.build(m, f'robot_{side}_', side); wc = np.array([rig.wrist_qpos[k] for k in ('x', 'y', 'z', 'rx', 'ry', 'rz')])
    oc = int(m.joint(obj + '_free').qposadr[0]); Tn = to_T(new[:, oc:oc + 3], new[:, oc + 3:oc + 7]); i = on.index(obj)
    To = to_T(tr['object_positions'][:n, i], tr['object_quaternions_wxyz'][:n, i]); To = np.concatenate([To, np.repeat(To[-1:], T - n, 0)])
    if obj in a.twin: To = To @ GTW
    D = np.einsum('tij,tjk->tik', Tn, np.linalg.inv(To))
    if obj in PRE:   # physical pre-contact delta (settled object in the new scene), blended into D after T0 over B seconds
        trj, fr, t0, bl, *post = PRE[obj]; z = np.load(trj); j = list(z['object_names']).index(obj); Ts = to_T(z['object_positions'][fr:fr + 1, j], z['object_quaternions_wxyz'][fr:fr + 1, j])[0]
        Dp = Ts @ np.linalg.inv(To[0]); w = sm((np.arange(T) / fps - t0) / bl)
        if post: w = w * (1 - (1 - post[2]) * sm((np.arange(T) / fps - post[0]) / post[1]))   # late blend back toward the pre delta (weight G on the compose delta)
        fr_ = dict(x.split(':') for x in a.pre_rot_scale).get(obj); fr_ = float(fr_) if fr_ is not None else 1.
        Rb = R.from_matrix(Dp[:3, :3]) * R.from_rotvec(fr_ * w[:, None] * (R.from_matrix(Dp[:3, :3]).inv() * R.from_matrix(D[:, :3, :3])).as_rotvec())
        po = To[:, :3, 3]; ppre = po @ Dp[:3, :3].T + Dp[:3, 3]; pn = Tn[:, :3, 3]; tgt = (1 - w[:, None]) * ppre + w[:, None] * pn
        Db = np.tile(np.eye(4), (T, 1, 1)); Db[:, :3, :3] = Rb.as_matrix(); Db[:, :3, 3] = tgt - np.einsum('tij,tj->ti', Db[:, :3, :3], po); D = Db; Tn = np.einsum('tij,tjk->tik', D, To)
    rv = R.from_matrix(D[:, :3, :3]).as_rotvec(); tv = D[:, :3, 3].copy()
    c = Tn[:, :3, 3]
    # D maps old object point q -> R q + tv. The old object centre co = R^T (c - tv) maps to c. Rotation about c: x' = c + R(x - co).
    co = np.einsum('tji,tj->ti', D[:, :3, :3], c - tv); disp = c - co
    if a.sigma > 0: rv = gaussian_filter1d(rv, a.sigma, axis=0, mode='nearest'); disp = gaussian_filter1d(disp, a.sigma, axis=0, mode='nearest'); co = gaussian_filter1d(co, a.sigma, axis=0, mode='nearest')
    g = np.clip((np.arange(T) / fps - a.t_from) * 1e9, 0, 1) * a.gain if a.t_from > 0 else np.full(T, a.gain)
    for k in range(T):
        Rc = R.from_rotvec(g[k] * rv[k]); p = Q[k, wc[:3]].copy(); ang = Q[k, wc[3:]].copy()
        Q[k, wc[:3]] = co[k] + g[k] * disp[k] + Rc.apply(p - co[k]); Q[k, wc[3:]] = continuous_euler_xyz(R.from_matrix((Rc * R.from_euler('XYZ', ang)).as_matrix()[None]), ang)[0]
    rec[obj] = dict(side=side, delta_rot_deg_mean=float(np.degrees(np.linalg.norm(rv, axis=1)).mean()), delta_disp_cm_mean=float(np.linalg.norm(disp, axis=1).mean() * 100))
for lf in a.lift:
    side, dz, l0, l1 = lf.split(':'); dz, l0, l1 = float(dz), float(l0), float(l1); rig = H.HandRig.build(m, f'robot_{side}_', side); tt = np.arange(T) / fps
    Q[:, rig.wrist_qpos['z']] += dz * sm((tt - l0) / 0.3) * (1 - sm((tt - l1) / 0.3))
for sq in a.squeeze:
    side, amt, s0, s1 = sq.split(':'); amt, s0, s1 = float(amt), float(s0), float(s1); rig = H.HandRig.build(m, f'robot_{side}_', side); tt = np.arange(T) / fps
    gq = sm((tt - s0) / 0.3) * (1 - sm((tt - s1) / 0.3))
    for jn_, adr in rig.finger_qpos.items():
        if jn_.endswith(('MCP_FE', 'PIP', 'DIP', 'thumb_IP', 'CMC_FE')): Q[:, adr] += amt * gq
lim = m.jnt_range[m.actuator_trnid[:, 0]]; qa = m.jnt_qposadr[m.actuator_trnid[:, 0]]; Q[:, qa] = np.clip(Q[:, qa], lim[:, 0], lim[:, 1])
out.mkdir(parents=True, exist_ok=False)
if a.gain == 0 and not a.lift and not a.squeeze: Q = ref['qpos'].copy()
refp = TR.save_reference(out / 'reference.npz', m, Q, ref['labels'], fps, ref['segment'], ref['segments'], source='object-relative wrist port to the compose targets (stream F v22, target_final_v5, twin sweeper)', arms_coupled=False, floating_runtime=True)
t0_ = __import__('time').time(); ex = execute_reference(a.scene, refp, out / 'execution'); man = json.loads((epd / 'input_manifest.json').read_text())
met = evaluate_files(man['source'], ex['trajectory'], epd / 'input_manifest.json', ex['scene'], epd / 'metric_geometry.json')
res = {k: v for k, v in prev.items() if k != 'ilc'}; res.update(twin=a.twin, reference=str(refp), execution=ex, metrics=met, input_dir=str(epd), source=str(src), port=dict(lift=a.lift, pre_rot_scale=a.pre_rot_scale, squeeze=a.squeeze, pre=a.pre, pairs=a.pair, sigma=a.sigma, gain=a.gain, t_from=a.t_from, deltas=rec), note='object-relative wrist port (stream F v22, target_final_v5, twin sweeper)')
(out / 'result.json').write_text(json.dumps(res, indent=1, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)) + '\n')
o = met['official_style_reference_proxy']['objects']; cs, e = compose_score_traj(ex['trajectory'])
print(json.dumps(dict(out=str(out), mp_sr=met['reference_proxy_mp_sr'], vs_ref={k: (round(v['mean_position_error_cm'][0], 2), round(v['mean_orientation_error_degrees'][0], 1)) for k, v in o.items()}, vs_compose=cs, deltas=rec, completed=ex['execution_completed'])), flush=True)
for s in range(0, 235, 20): print(' t', s / 20, [(round(e[k]['cm_t'][s], 1), round(e[k]['deg_t'][s])) for k in NAMES])
