"""Fixed-grip planner: a stand-in for goal-and-plan agents, not DexAgent itself. It is given the pen's goal path, and also generous advantages: a floating wrist, and a physics-checked search over grips.
For each grip it closes the fingers around a frozen pen, releases, checks the hold, then moves the wrist so the pen follows the path with fingers fixed. servo=1 corrects the wrist from the measured pen error (a replanning agent).
Usage: planner.py OUT.jsonl TRAJ_DIR"""
import sys, json, numpy as np, mujoco, multiprocessing as mp, os
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wmetric as W
XML = str(W.P / 'scene.xml'); DT_REF = .02; GN = None
_m = None
def T(p, q_wxyz):
    X = np.eye(4); X[:3, :3] = R.from_quat(np.asarray(q_wxyz)[[1, 2, 3, 0]]).as_matrix(); X[:3, 3] = p; return X
def pen_T(q): return T(q[:3], q[3:])
def setup():
    m = mujoco.MjModel.from_xml_path(XML)
    if os.environ.get('REAL'):  # official Sharpa finger actuators (kp, damping, armature, joint friction) in place of SPIDER's kp 300 and armature 1
        O = json.load(open(W.H / 'official_act.json')); Om = mujoco.MjModel.from_xml_path('/scratch2/exaFLOPs/SRL_V2D/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/xmls/sharpawave/right_sharpawave.xml')
        for a_ in range(6, m.nu):
            jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[a_, 0]); g, bp, _ = O[jn]; m.actuator_gainprm[a_, 0] = g; m.actuator_biasprm[a_, :3] = bp
            oj = mujoco.mj_name2id(Om, mujoco.mjtObj.mjOBJ_JOINT, jn); dd = Om.jnt_dofadr[oj]; sd = m.jnt_dofadr[m.actuator_trnid[a_, 0]]; m.dof_armature[sd] = Om.dof_armature[dd]; m.dof_damping[sd] = Om.dof_damping[dd]; m.dof_frictionloss[sd] = Om.dof_frictionloss[dd]
    d = mujoco.MjData(m); b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'right_hand_C_MC'); return m, d, b
def wrist_T(m, d, b, q):
    d.qpos[:] = q; mujoco.mj_kinematics(m, d); X = np.eye(4); X[:3, :3] = d.xmat[b].reshape(3, 3); X[:3, 3] = d.xpos[b]; return X
def base_from(m, d, b, Xw, qref):
    """Six base joints that put the hand link at Xw, by Gauss-Newton on FK starting at qref."""
    q = qref.copy(); jac = np.zeros((6, m.nv)); 
    for _ in range(25):
        X = wrist_T(m, d, b, q); e = np.r_[Xw[:3, 3] - X[:3, 3], R.from_matrix(Xw[:3, :3] @ X[:3, :3].T).as_rotvec()]
        if np.linalg.norm(e) < 1e-6: break
        mujoco.mj_forward(m, d); mujoco.mj_jacBody(m, d, jac[:3], jac[3:], b); q[:6] += np.linalg.solve(jac[:, :6] + 1e-9 * np.eye(6), e)
    return q[:6]
def rollout(a):
    f, squeeze, servo, press, seed = a; m, d, b = setup(); rng = np.random.default_rng(seed); RF = W.RF; k = RF['q']; substeps = int(round(DT_REF / m.opt.timestep))
    pen_id = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'right_object_joint')]; dof = m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'right_object_joint')]
    X0 = np.linalg.inv(wrist_T(m, d, b, k[f])) @ pen_T(k[f, -7:])  # pen in hand at the grip frame
    if seed:  # jitter the pen in the hand: 4 mm and 0.1 rad, the size that can open a new grip
        J = np.eye(4); J[:3, :3] = R.from_rotvec(rng.normal(0, .1, 3)).as_matrix(); J[:3, 3] = rng.normal(0, .004, 3); X0 = J @ X0
    if GN:  # execution error when a real system places the grip: fresh noise on top of the chosen grip
        r2 = np.random.default_rng(GN[2]); J = np.eye(4); J[:3, :3] = R.from_rotvec(r2.normal(0, GN[1], 3)).as_matrix(); J[:3, 3] = r2.normal(0, GN[0], 3); X0 = J @ X0
    qf = k[f, 6:28].copy(); lo, hi = m.actuator_ctrlrange[6:, 0], m.actuator_ctrlrange[6:, 1]
    FLEX = np.array([('FE' in mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or 'PIP' in mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or 'DIP' in mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or 'IP' in mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)) for j in range(6, 28)], float)
    goal = np.clip(qf + squeeze * FLEX, lo, hi); opn = np.clip(qf * float(os.environ.get('OPEN', .45)), lo, hi)
    pen0 = k[0, -7:].copy(); Xw0 = pen_T(pen0) @ np.linalg.inv(X0); base = base_from(m, d, b, Xw0, k[0])
    mujoco.mj_resetData(m, d); d.qpos[:6] = base; d.qpos[6:28] = opn; d.qpos[pen_id:pen_id + 7] = pen0; d.ctrl[:6] = base; d.ctrl[6:] = opn; mujoco.mj_forward(m, d)
    # Close around the frozen pen, then release and hold.
    nc = int(.8 / m.opt.timestep)
    for i in range(nc):
        d.ctrl[6:] = opn + (goal - opn) * min(1., i / (.6 / m.opt.timestep)); d.qpos[pen_id:pen_id + 7] = pen0; d.qvel[dof:dof + 6] = 0; d.qfrc_applied[:6] = d.qfrc_bias[:6]; mujoco.mj_step(m, d)
    for i in range(int(.7 / m.opt.timestep)): d.qfrc_applied[:6] = d.qfrc_bias[:6]; mujoco.mj_step(m, d)
    if np.isnan(d.qpos).any(): return dict(f=f, squeeze=squeeze, servo=servo, hold_cm=999., PASS=False, fail='nan')
    p = d.qpos[pen_id:pen_id + 7].copy(); hold = float(np.linalg.norm(p[:3] - pen0[:3])); ang = float((R.from_quat(p[3:][[1, 2, 3, 0]]) * R.from_quat(pen0[3:][[1, 2, 3, 0]]).inv()).magnitude())
    res = dict(f=int(f), squeeze=float(squeeze), servo=float(servo), press=float(press), seed=int(seed), hold_cm=round(hold * 100, 2), hold_rad=round(ang, 3))
    if hold > .03 or ang > .5: return dict(res, PASS=False, fail='grip does not hold the pen')
    Xh = np.linalg.inv(wrist_T(m, d, b, d.qpos.copy())) @ pen_T(p); mujoco.mj_forward(m, d)  # relation actually achieved
    QS, TAU, PEN, NC = [], [], [], []; PG = {i for i in range(m.ngeom) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or '').startswith('right_object_') and 'visual' not in mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i)}; BG = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'board')
    prs = np.convolve(RF['on'].astype(float), np.ones(25) / 25, 'same'); C = np.eye(4); out = []; q0 = d.qpos.copy(); mjq = q0.copy(); 
    for t in range(RF['n']):
        Xp = pen_T(RF['pen'][t]); cur = pen_T(d.qpos[pen_id:pen_id + 7])
        if press: Xp[:3, 3] -= W.N * press * prs[t]  # ask for the tip slightly inside the board while writing, to keep contact; the score still uses the true path
        if servo:  # slow integral correction of the pen pose error, a replanning stand-in
            E = Xp @ np.linalg.inv(cur); dE = np.eye(4); rv = R.from_matrix(E[:3, :3]).as_rotvec() * servo * .5; dE[:3, :3] = R.from_rotvec(np.clip(rv, -.05, .05)).as_matrix(); dE[:3, 3] = np.clip(E[:3, 3] * servo, -.01, .01); C = dE @ C
        Xw = C @ Xp @ np.linalg.inv(Xh)
        mjd = mujoco.MjData(m) if t == 0 else mjd
        mjd.qpos[:] = d.qpos; base = base_from(m, mjd, b, Xw, d.qpos.copy()); d.ctrl[:6] = base
        for _ in range(substeps): d.qfrc_applied[:6] = d.qfrc_bias[:6]; mujoco.mj_step(m, d)  # a real arm carries the hand's weight
        if np.isnan(d.qpos).any(): return dict(res, PASS=False, fail='nan')
        out.append(d.qpos[pen_id:pen_id + 7].copy()); QS.append(d.qpos.copy()); TAU.append(np.abs(d.actuator_force[6:]).copy())
        if t % 5 == 0:
            ds = [d.contact[i].dist for i in range(d.ncon) if (d.contact[i].geom1 in PG) != (d.contact[i].geom2 in PG) and BG not in (d.contact[i].geom1, d.contact[i].geom2)]; PEN.append(min(ds) if ds else np.nan); NC.append(len(ds))
    s = W.score(np.array(out), f'planner f{f} s{squeeze} v{servo} p{press} j{seed}'); extra = dict(pen_depth_cm=round(float(-np.nanpercentile(PEN, 5)) * 100, 2), contacts=round(float(np.nanmedian(NC)), 1), tau_med=round(float(np.median(np.max(TAU, 1))), 2), tau_p99=round(float(np.percentile(np.max(TAU, 1), 99)), 2))
    if os.environ.get('REC'): np.savez(os.environ['REC'], q=np.array(QS), pen=np.array(out), tau=np.array(TAU), q_settled=q0, Xh=Xh)
    return dict(res, **s, **extra, q_goal=goal.tolist())
if __name__ == '__main__':
    out = sys.argv[1]; rng = np.random.default_rng(7); N = int(os.environ.get('NJOBS', 6000)); frames = np.arange(0, W.RF['n'], 8); sq = [0., .02, .05, .1, .2, .35, .5, .8]; sv = [0., .02, .05, .1, .2, .4]; pr = [0., .001, .002, .004, .007]
    jobs = [(0, 0., 0., 0., 0)] if os.environ.get('SMOKE') else [(int(rng.choice(frames)), float(rng.choice(sq)), float(rng.choice(sv)), float(rng.choice(pr)), i + 1) for i in range(N)]
    print('jobs', len(jobs), flush=True)
    with mp.Pool(int(os.environ.get('SLURM_CPUS_PER_TASK', 2))) as pool, open(out, 'a') as fo:
        for r in pool.imap_unordered(rollout, jobs): fo.write(json.dumps(r) + '\n'); fo.flush(); print({k: v for k, v in r.items() if k != 'q_goal'}, flush=True)
