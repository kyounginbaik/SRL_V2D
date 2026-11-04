"""Stream P ep33 phase 6: re-target an existing registered actor to the compose-world reference by wrist corrections folded into its
phase weights (same mechanism as wrist_refit.py / rollplan.py: architecture, feedback, margins and releases unchanged).
For each hand/object pair (right/wooden_spoon, left/blue_cup), from a base rollout A(t) of the actor (any world; normally the v21
actor in the old world, where the grasp works) the wrist command is corrected by C(t) = S*(t) A(t)^-1 (pivot at the achieved object)
with the target object pose
   S*(t) = blend( T_d A(t),  N(t),  b(t) ),   T_d = N_rest O_rest^-1 (resting object pose in the new-world port rollout vs the old-world base rollout, before the hand touches it: the hand approaches the new object
   exactly as it approached the old one), N = new reference (setup frame), b(t) = gain * smoothstep ramp on [t_b0, t_b1] and down
   on [t_e0, t_e1] (blend back to T_d A, i.e. the old relative placement, e.g. for the release); position and rotation blended
   (slerp); for the cup only the axis-aligning (minimal) rotation is used (axially symmetric).
usage: track_refit.py BASE_DIR OUT_DIR [--right t_b0,t_b1,t_e0,t_e1,gain_pos,gain_rot] [--left ...] [--base-world old|new] [--no-eval]"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(k, '1')
import sys, json, hashlib, argparse, numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R, Slerp
B = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017'); sys.path[:0] = [str(B / 'code'), '/data1/SRL_V2D/third_party/DexAgent']
from wrist_reference import continuous_euler_xyz
from learned_feedback_policy import features
P = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamP_ep33'); SC = P / os.environ.get('CW_OUT', 'staging_C') / 'scene/scene_physics.xml'; OEP = P / os.environ.get('CW_OUT', 'staging_C') / 'episode_compose/episode_000033'
CUPAX = np.array([-0.0237963577378219, -0.005046973670649213, 0.9997040869253169])
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
sm = lambda u: np.clip(u, 0, 1) ** 3 * (10 - 15 * np.clip(u, 0, 1) + 6 * np.clip(u, 0, 1) ** 2)
ap = argparse.ArgumentParser(); ap.add_argument('base'); ap.add_argument('out'); ap.add_argument('--right'); ap.add_argument('--left'); ap.add_argument('--ridge', type=float, default=1e-3)
ap.add_argument('--new-rollout', default=str(P / 'C/port_cuptilt/evaluation/execution/trajectory.npz'), help='rollout of the ported actor in the NEW world: resting object poses for T_d')
ap.add_argument('--rest-frames', default='12,34', help='frames (cup, spoon) at which both objects rest untouched in both rollouts')
ap.add_argument('--held-old', help='right hand only: old-world rollout of the same base actor; after --grasp-frame the held spoon is estimated as Delta A_old(t), Delta = A(f_g) A_old(f_g)^-1 (the base rollout may lose the spoon later)')
ap.add_argument('--grasp-frame', type=int, default=62)
ap.add_argument('--follow-cup', default='', help='right wrist translated by g * (cup_new(t) - cup_old(t)) on [t0, t1] (0.3 s ramps): keeps the spoon-in-cup geometry when the cup moves to the new reference; g,t0,t1; cup_new from --new-rollout, cup_old from the base rollout')
ap.add_argument('--squeeze', default='', help='right pinch closing bias: S_rad,t0,t1 (thumb CMC_FE/MCP_FE/IP, index PIP/DIP; 0.25 s ramps)')
ap.add_argument('--close3', default='', help='right middle/ring/pinky closing bias: C_rad,t0,t1')
ap.add_argument('--td-frac', type=float, default=1.0, help='right hand: fraction of the resting-spoon offset T_d (rotation + translation)')
ap.add_argument('--td-fade', default='', help='right hand: t0,t1 [s] fade T_d out (back to the unshifted command) after the grasp')
ap.add_argument('--rollout', help='trajectory.npz of the base rollout to use for A(t) and the commands (default: the base evaluation)'); ap.add_argument('--no-eval', action='store_true')
a = ap.parse_args(); base, out = Path(a.base).resolve(), Path(a.out).resolve()
if (out / 'training/policy.npz').exists(): raise SystemExit('exists ' + str(out))
ev = json.loads((base / 'evaluation/result.json').read_text()); trp = a.rollout or ev['execution']['trajectory']; tr = np.load(trp)
man_old = json.loads((Path(ev['input_dir']) / 'input_manifest.json').read_text()); S = np.asarray(man_old['setup_from_source'])
Wold = S @ np.load(Path(man_old['source']) / 'object_poses.npz')['world_from_object']
man = json.loads((OEP / 'input_manifest.json').read_text()); Wnew = np.asarray(man['setup_from_source']) @ np.load(Path(man['source']) / 'object_poses.npz')['world_from_object']
n = 348; ft = np.arange(n) / 20.0; st = tr['substep_time_s']; sc = tr['substep_ctrl']; names = [str(x) for x in tr['object_names']]
with np.load(base / 'training/policy.npz', allow_pickle=False) as z: d = {k: z[k] for k in z.files}
d['scene_sha256'] = np.array(sha(SC)); na = [str(x) for x in d['actuator_names']]
def axis_rot(Ra, Rb):   # minimal rotation taking the cup axis of Ra onto that of Rb (world)
    u, v = Ra @ CUPAX, Rb @ CUPAX; c = np.cross(u, v); s = np.linalg.norm(c); ang = np.arctan2(s, np.dot(u, v)); return R.from_rotvec(c / s * ang if s > 1e-9 else np.zeros(3)).as_matrix()
allids, cols, info = [], [], {}
for side, obj, spec in (('right', 'wooden_spoon', a.right), ('left', 'blue_cup', a.left)):
    if not spec: continue
    tb0, tb1, te0, te1, gp, gr = [float(x) for x in spec.split(',')]; i = names.index(obj); oi = 1 if obj == 'wooden_spoon' else 0
    Ap = tr['object_positions'][:n, i].copy(); AR = R.from_quat(tr['object_quaternions_wxyz'][:n, i][:, [1, 2, 3, 0]]).as_matrix()
    if obj == 'wooden_spoon' and a.held_old:
        to = np.load(a.held_old); io = [str(x) for x in to['object_names']].index(obj); fg = a.grasp_frame
        Op = to['object_positions'][:n, io]; OR = R.from_quat(to['object_quaternions_wxyz'][:n, io][:, [1, 2, 3, 0]]).as_matrix()
        Dr = AR[fg] @ OR[fg].T
        for f in range(fg + 1, n): Ap[f] = Ap[fg] + Dr @ (Op[f] - Op[fg]); AR[f] = Dr @ OR[f]
    fr = int(a.rest_frames.split(',')[0 if obj == 'blue_cup' else 1]); tn = np.load(a.new_rollout); jn_ = [str(x) for x in tn['object_names']].index(obj)
    mk = lambda pp, qq: np.vstack([np.c_[R.from_quat(qq[[1, 2, 3, 0]]).as_matrix(), pp], [0, 0, 0, 1]])
    O0 = mk(Ap[fr], tr['object_quaternions_wxyz'][fr, i]); N0 = mk(tn['object_positions'][fr, jn_], tn['object_quaternions_wxyz'][fr, jn_])   # resting poses (old world rollout / new world rollout)
    if obj == 'blue_cup': Rd = axis_rot(O0[:3, :3], N0[:3, :3])
    else:
        Rd = R.from_rotvec(a.td_frac * R.from_matrix(N0[:3, :3] @ O0[:3, :3].T).as_rotvec()).as_matrix()
        N0 = N0.copy(); N0[:3, 3] = O0[:3, 3] + a.td_frac * (N0[:3, 3] - O0[:3, 3])
    # T_d about the old initial object origin: x -> N0.p + Rd (x - O0.p)
    tgt_p, tgt_R = np.zeros((n, 3)), np.zeros((n, 3, 3))
    for f in range(n):
        fade = 1.0
        if obj == 'wooden_spoon' and a.td_fade:
            f0_, f1_ = [float(x) for x in a.td_fade.split(',')]; fade = 1 - sm((ft[f] - f0_) / max(f1_ - f0_, 1e-3))
        Rdf = R.from_rotvec(fade * R.from_matrix(Rd).as_rotvec()).as_matrix(); tdf = fade * (N0[:3, 3] - O0[:3, 3])
        p1 = O0[:3, 3] + tdf + Rdf @ (Ap[f] - O0[:3, 3]); R1 = Rdf @ AR[f]
        if obj == 'blue_cup': R2 = axis_rot(R1, Wnew[f, oi, :3, :3]) @ R1
        else: R2 = Wnew[f, oi, :3, :3]
        b = sm((ft[f] - tb0) / max(tb1 - tb0, 1e-3)) * (1 - sm((ft[f] - te0) / max(te1 - te0, 1e-3)))
        tgt_p[f] = p1 + gp * b * (Wnew[f, oi, :3, 3] - p1)
        tgt_R[f] = (R.from_matrix(R1) * R.from_rotvec(gr * b * (R.from_matrix(R1).inv() * R.from_matrix(R2)).as_rotvec())).as_matrix()
    Cr = R.from_matrix(np.einsum('nij,nkj->nik', tgt_R, AR))   # world rotation taking achieved -> target
    Crs = Slerp(ft, Cr)(np.clip(st, 0, ft[-1])); aps = np.stack([np.interp(st, ft, Ap[:, k]) for k in range(3)], 1); tps = np.stack([np.interp(st, ft, tgt_p[:, k]) for k in range(3)], 1)
    ids = [na.index(f'robot_{side}_{side}_{k}_position') for k in ('pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z')]
    p0 = sc[:, ids[:3]]; ang0 = sc[:, ids[3:]]; p1 = tps + Crs.apply(p0 - aps); R1 = Crs * R.from_euler('XYZ', ang0); ang1 = ang0.copy()
    for k in range(len(st)): ang1[k] = continuous_euler_xyz(R.from_matrix(R1[k].as_matrix()[None]), ang1[k - 1] + (ang0[k] - ang0[k - 1]) if k else ang0[k])[0]
    D = np.concatenate([p1 - p0, ang1 - ang0], 1)
    selfg = np.array([d['feedback_weights'][j, j] / d['state_scale'][j] * d['action_scale'][j] for j in ids]); D = D * (1 - selfg)
    allids += ids; cols.append(D); spd = np.abs(np.diff(ang1, axis=0) / np.diff(st)[:, None]).max()
    info[side] = dict(spec=spec, Td_rot_deg=float(np.degrees(R.from_matrix(Rd).magnitude())), Td_trans_cm=float(np.linalg.norm(N0[:3, 3] - O0[:3, 3]) * 100), max_pos_change_m=float(np.abs(D[:, :3]).max()), max_rot_change_rad=float(np.abs(D[:, 3:]).max()), max_rot_joint_speed=float(spd))
if a.follow_cup:
    g_, c0, c1 = [float(x) for x in a.follow_cup.split(',')]; tn2 = np.load(a.new_rollout); ic = [str(x) for x in tn2['object_names']].index('blue_cup')
    dc = tn2['object_positions'][:n, ic] - tr['object_positions'][:n, names.index('blue_cup')]
    from scipy.ndimage import gaussian_filter1d
    dc = gaussian_filter1d(dc, 3, axis=0); gq = sm((st - c0) / 0.3) * (1 - sm((st - (c1 - 0.3)) / 0.3))
    ids_r = [na.index(f'robot_right_right_{k}_position') for k in ('pos_x', 'pos_y', 'pos_z')]
    dcs = np.stack([np.interp(st, ft, dc[:, k]) for k in range(3)], 1) * (g_ * gq)[:, None]
    for k, j in enumerate(ids_r):
        if j in allids: cols[0][:, allids.index(j)] += dcs[:, k] * (1 - d['feedback_weights'][j, j] / d['state_scale'][j] * d['action_scale'][j])
        else: allids.append(j); cols.append((dcs[:, k] * (1 - d['feedback_weights'][j, j] / d['state_scale'][j] * d['action_scale'][j]))[:, None])
    info['follow_cup'] = a.follow_cup
for spec, joints in ((a.squeeze, [f'thumb_{j}' for j in ('CMC_FE', 'MCP_FE', 'IP')] + ['index_PIP', 'index_DIP']), (a.close3, [f'{f}_{j}' for f in ('middle', 'ring', 'pinky') for j in ('MCP_FE', 'PIP', 'DIP')])):
    if not spec: continue
    amt, q0, q1 = [float(x) for x in spec.split(',')]; gq = sm((st - q0) / 0.25) * (1 - sm((st - (q1 - 0.25)) / 0.25))
    for jn in joints:
        j = na.index(f'robot_right_right_{jn}_position'); dj = np.clip(sc[:, j] + amt * gq, d['ctrl_lo'][j], d['ctrl_hi'][j]) - sc[:, j]
        allids.append(j); cols.append((dj * (1 - d['feedback_weights'][j, j] / d['state_scale'][j] * d['action_scale'][j]))[:, None])
    info['fingers_' + ('pinch' if 'thumb' in joints[0] else 'wrap3')] = spec
Dall = np.concatenate(cols, 1); Phi = features(st / float(d['duration_s']), d['centers'], float(d['width'])); target = Dall / d['action_scale'][allids]
Dw = np.linalg.solve(Phi.T @ Phi + a.ridge * np.eye(Phi.shape[1]), Phi.T @ target); W = d['phase_weights'].copy(); W[:, allids] += Dw; d['phase_weights'] = W
perr = float(np.abs(Phi @ Dw * d['action_scale'][allids] - Dall).max())
(out / 'training').mkdir(parents=True, exist_ok=True); ck = out / 'training/policy.npz'; np.savez_compressed(ck, **d)
rep = json.loads((base / 'training/training_report.json').read_text()); old = {k: rep.get(k) for k in ('scene', 'scene_sha256')}
rep.update(scene=str(SC), scene_sha256=sha(SC), checkpoint_sha256=sha(ck), compose_track_refit=dict(base_actor=str(base / 'training/policy.npz'), base_sha256=sha(base / 'training/policy.npz'), base_rollout=trp, base_world_fields=old,
    new_reference=str(Path(man['source']) / 'object_poses.npz'), new_reference_sha256=man['source_object_poses_sha256'], hands=info, td_frac=a.td_frac, td_fade=a.td_fade, held_old=a.held_old, grasp_frame=a.grasp_frame, projection_max_error=perr, script=str(Path(__file__).resolve()), script_sha256=sha(__file__)))
(out / 'training/training_report.json').write_text(json.dumps(rep, indent=2, default=str))
print('refit', json.dumps(info), 'proj', round(perr, 4), flush=True)
if a.no_eval: sys.exit()
from evaluate_policy_component import evaluate
r = evaluate(33, str(SC), str(ck), str(out / 'evaluation'), str(OEP)); o = r['metrics']['official_style_reference_proxy']['objects']
t2 = np.load(r['execution']['trajectory']); pe = np.linalg.norm(t2['object_positions'][:, 1] - Wnew[:, 1, :3, 3], axis=1) * 100
print('DONE', out.name, json.dumps(dict(mp_sr=r['metrics']['reference_proxy_mp_sr'], objects={k: (round(v['mean_position_error_cm'][0], 3), round(v['mean_orientation_error_degrees'][0], 2), round(v['max_position_error_m'][0] * 100, 1)) for k, v in o.items()},
    spoon_err_cm_by_2s=[round(float(pe[s:s + 40].mean()), 1) for s in range(0, n, 40)])), flush=True)
