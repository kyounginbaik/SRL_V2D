"""T33: move the LEFT-hand cup grasp to the cup's image-left side (as the human, phi ~ -100 deg) and firm up the RIGHT-hand spoon grasp,
by editing an open-loop teacher reference (servo targets, streamB format: reference.npz + result.json + execution/).
Left: every left-wrist command is rotated about the EXECUTED cup symmetry axis (line through the cup COM, base rollout) by Delta(t):
  W'(t) = C(t) Rz_axis(Delta(t)) C(t)^-1 W(t). With a rigid grasp the cup pose only spins about its own symmetry axis (scoring-invariant).
  mode const: Delta(t) = D * ramp(t); mode follow: Delta(t) = smooth(phi_goal - phi_base(t)) * ramp(t) (left palm kept at phi_goal).
  ramp = smoothstep over [ta, tb] (before the left hand touches the cup; the cup is static then, so the hand only orbits the axisymmetric cup).
  After t_rel the cup pose is frozen at t_rel (release / place).
Right: non-thumb flexion + rsq rad, thumb flexion + rsqt rad over [r0, r1] (0.3 s ramps); optionally only some fingers (--rfing).
Finger commands of the left hand are unchanged (same grasp, other side).
usage: lside33.py grid BASE OUTROOT --variants JSON  |  lside33.py make BASE OUT [--D ..|--follow PHI] [--rsq ..] ...  (make = execute_reference + score)"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import sys, json, argparse, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np, mujoco
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import gaussian_filter1d
sys.path.insert(0, str(Path(__file__).resolve().parent)); import t33lib as T
from tool_library.common import tracking as TR
from floating_runtime import execute_reference, servo_rollout
from wrist_reference import continuous_euler_xyz
sm = lambda u: np.clip(u, 0, 1) ** 3 * (10 - 15 * np.clip(u, 0, 1) + 6 * np.clip(u, 0, 1) ** 2)
DEF = dict(D=-120.0, follow=None, ta=0.0, tb=0.6, t_rel=16.4, rsq=0.0, rsqt=0.0, r0=2.2, r1=14.6, rfing='index,middle,ring,pinky', lsq=0.0, lt0=None, sigma=0.5,
           lilc=0, lgp=0.5, lgr=0.6, l0=1.0, l1=16.4, lsig=0.3, rflex={}, rf0=2.0, rfr=0.5, rf1=14.6, raa={}, ikg=None)
# ikg: blend the RIGHT hand towards the human Mink-IK grip (W1 source sharpa_ik.npz, wxyz) taken at frame `frame`: wrist target
#   T_spoon_exec(t) @ local_IK (local wrist pose relative to the spoon in the IK file), fingers -> lo + relax * (IK - lo) on flexion joints
#   (AA as IK); weight alpha * smoothstep over [t0, t0 + ramp], back to the base commands over [t1 - 0.3, t1].
#   dict(alpha=1.0, t0=2.6, ramp=1.0, relax=1.0, frame=100, t1=14.6)
# rflex: per right finger flexion offset (rad, added to every non-AA joint of that finger) ramped in over [rf0, rf0 + rfr], out at rf1;
# raa: per right finger MCP_AA offset (rad), same ramp. D=0 leaves the left hand unchanged.
def noise_draws(m, n, level, N, seed0=777):
    """team fcem_r.py robustness noise: all robot command columns, N(0,1) smoothed (sigma 5 frames) * level * 3.5, frame 0 zero."""
    rob = [i for i in range(m.njnt) if (m.joint(i).name or '').startswith('robot_')]; cols = [int(m.jnt_qposadr[i]) for i in rob]; out = []
    for j in range(N):
        g = np.random.default_rng(seed0 + j); Z = np.zeros((n, m.nq)); Z[:, cols] = gaussian_filter1d(g.normal(size=(n, len(cols))), 5, axis=0) * level * 3.5; Z[0] = 0; out.append(Z)
    return out
def drop_metrics(m, Qx, fps=20.0):
    """target-independent grasp tests: spoon dropped = right fingers on the spoon == 0 for >= 0.5 s inside [2.6, 14.0] s, or spoon origin -
    right palm distance > 20 cm (held: ~14 cm) for >= 0.25 s there; cup lost = left fingers 0 for >= 0.5 s in [1.5, 16.0], or cup COM - left
    palm > 22 cm for >= 0.25 s. passed = both objects < 3 cm / 30 deg vs $E31_TGT."""
    e = T.errors(m, Qx); S, C, _ = T.contacts(m, Qx); ns = np.array([len(x) for x in S]); nc = np.array([len(x) for x in C])
    d = mujoco.MjData(m); bs, bc, br, bl = (m.body(x).id for x in ('wooden_spoon', 'blue_cup', 'robot_right_right_hand_C_MC', 'robot_left_left_hand_C_MC')); ds, dc = np.zeros(len(Qx)), np.zeros(len(Qx))
    for k, q in enumerate(Qx):
        d.qpos[:] = q; mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d); ds[k] = np.linalg.norm(d.xpos[bs] - d.xpos[br]); dc[k] = np.linalg.norm(d.xipos[bc] - d.xpos[bl])
    def run(z):
        best = cur = 0
        for v in z: cur = cur + 1 if v else 0; best = max(best, cur)
        return best / fps
    k0, k1 = int(2.6 * fps), int(14.0 * fps); c0, c1 = int(1.5 * fps), int(16.0 * fps); res = T.summary(e)
    return dict(summary=res, passed=bool(all(p <= 3 and d_ <= 30 for p, d_ in res.values())),
                spoon_drop=bool(run(ns[k0:k1] == 0) >= 0.5 or run(ds[k0:k1] > 0.20) >= 0.25), cup_lost=bool(run(nc[c0:c1] == 0) >= 0.5 or run(dc[c0:c1] > 0.22) >= 0.25),
                r_fingers=float(ns[k0:k1].mean()), spoon_palm_max_cm=float(100 * ds[k0:k1].max()), cup_palm_max_cm=float(100 * dc[c0:c1].max()))
EPD = Path('/data1/hansol/teacher33/worlds/R33c18_W1/ep33/episode_corrected/episode_000033')   # c18 target in the W1 physics world (mkref)

class Base:
    def __init__(self, base, scene=None):
        self.base = Path(base); r = json.loads((self.base / 'result.json').read_text()); self.scene = scene or r['execution']['scene']
        self.m = m = mujoco.MjModel.from_xml_path(self.scene); self.ref = TR.load_reference(self.base / 'reference.npz', m); self.Q0 = self.ref['qpos'].copy()
        self.n = len(self.Q0); self.fps = self.ref['fps']; self.t = np.arange(self.n) / self.fps
        tr = np.load(r['execution']['trajectory']); Qx = tr['qpos'][:self.n]; self.Qx = Qx
        ca = int(m.joint('blue_cup_free').qposadr[0]); ip = m.body_ipos[m.body('blue_cup').id]
        Rc = R.from_quat(Qx[:, ca + 3:ca + 7], scalar_first=True); self.cen = Qx[:, ca:ca + 3] + Rc.apply(ip); ax = Rc.apply(T.CUPAX)
        self.ax = ax * np.sign(ax[:, 2:3]); self.phi_base = T.azimuth(m, Qx)[:, 0]
        self.adl = [int(m.joint(f'robot_left_left_{j}').qposadr[0]) for j in ('pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z')]
        self.flex = {}
        for j in range(m.njnt):
            nm = m.joint(j).name
            for side in ('left', 'right'):
                for fi in T.FING:
                    if nm.startswith(f'robot_{side}_{side}_{fi}_') and not nm.endswith('_AA'):
                        self.flex.setdefault((side, fi), []).append((int(m.jnt_qposadr[j]), *m.jnt_range[j]))
    def commands(self, v):
        c = dict(DEF); c.update(v); Q = self.Q0.copy(); t = self.t; n = self.n; ramp = sm((t - c['ta']) / max(1e-3, c['tb'] - c['ta']))
        kr = min(n - 1, int(round(c['t_rel'] * self.fps))); cen = self.cen.copy(); ax = self.ax.copy(); cen[kr:] = cen[kr]; ax[kr:] = ax[kr]
        if c['follow'] is not None:
            dl = np.unwrap(np.radians(c['follow'] - self.phi_base)); dl[kr:] = dl[kr]; dl = gaussian_filter1d(dl, c['sigma'] * self.fps, mode='nearest')
        else: dl = np.full(n, np.radians(c['D']))
        dl = dl * ramp; a = self.adl
        for k in range(1, n):
            if dl[k] == 0: continue
            Rk = R.from_rotvec(ax[k] * dl[k]); p = Q[k, a[:3]]
            Q[k, a[:3]] = cen[k] + Rk.apply(p - cen[k])
            Q[k, a[3:]] = continuous_euler_xyz(R.from_matrix((Rk * R.from_euler('XYZ', Q[k, a[3:]])).as_matrix()[None]), Q[k - 1, a[3:]])[0]
        g = sm((t - c['r0']) / 0.3) * (1 - sm((t - (c['r1'] - 0.3)) / 0.3)); fs = [x for x in c['rfing'].split(',') if x]
        for fi in T.FING:
            amt = c['rsqt'] if fi == 'thumb' else (c['rsq'] if fi in fs else 0.0)
            if amt:
                for col, lo, hi in self.flex[('right', fi)]: Q[:, col] = np.clip(Q[:, col] + amt * g, lo, hi)
        gf = sm((t - c['rf0']) / max(1e-3, c['rfr'])) * (1 - sm((t - (c['rf1'] - 0.3)) / 0.3))
        for fi, amt in c['rflex'].items():
            for col, lo, hi in self.flex[('right', fi)]: Q[:, col] = np.clip(Q[:, col] + amt * gf, lo, hi)
        for fi, amt in c['raa'].items():
            j = self.m.joint(f'robot_right_right_{fi}_MCP_AA'); col = int(j.qposadr[0]); lo, hi = j.range; Q[:, col] = np.clip(Q[:, col] + amt * gf, lo, hi)
        if c['ikg']:
            Q = self.ik_blend(Q, dict(dict(alpha=1.0, t0=2.6, ramp=1.0, relax=1.0, frame=100, t1=14.6), **c['ikg']))
        if c['lsq']:
            l0 = c['tb'] + 0.3 if c['lt0'] is None else c['lt0']; gl = sm((t - l0) / 0.3) * (1 - sm((t - (c['t_rel'] - 0.3)) / 0.3))
            for fi in T.FING:
                for col, lo, hi in self.flex[('left', fi)]: Q[:, col] = np.clip(Q[:, col] + c['lsq'] * gl, lo, hi)
        Q[0] = self.Q0[0]; return Q
    def ik_blend(self, Q, g):
        m = self.m; n = self.n; t = self.t
        man = json.loads((T.W1 / 'input_manifest.json').read_text()); src = Path(man['source'])
        ik = np.load(src / 'sharpa_ik.npz', allow_pickle=True); op = np.load(src / 'object_poses.npz'); k = int(g['frame'])
        si = [str(x) for x in op['names']].index('wooden_spoon'); rq = ik['right_qpos'][k]; jn = [str(x) for x in ik['right_joint_names']]
        Tw = np.eye(4); Tw[:3, :3] = R.from_quat(rq[3:7], scalar_first=True).as_matrix(); Tw[:3, 3] = rq[:3]; loc = np.linalg.inv(op['world_from_object'][k, si]) @ Tw
        sa = int(m.joint('wooden_spoon_free').qposadr[0]); ps = self.Qx[:n, sa:sa + 3]; Rs = R.from_quat(self.Qx[:n, sa + 3:sa + 7], scalar_first=True)
        adr = [int(m.joint(f'robot_right_right_{j}').qposadr[0]) for j in ('pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z')]
        w = g['alpha'] * sm((t - g['t0']) / max(1e-3, g['ramp'])) * (1 - sm((t - (g['t1'] - 0.3)) / 0.3)); Q = Q.copy()
        for kk in range(1, n):
            if w[kk] <= 0: continue
            Rt = Rs[kk] * R.from_matrix(loc[:3, :3]); pt = ps[kk] + Rs[kk].apply(loc[:3, 3]); BR = R.from_euler('XYZ', Q[kk, adr[3:]])
            Q[kk, adr[:3]] = (1 - w[kk]) * Q[kk, adr[:3]] + w[kk] * pt; Rn = BR * R.from_rotvec((BR.inv() * Rt).as_rotvec() * w[kk])
            Q[kk, adr[3:]] = continuous_euler_xyz(R.from_matrix(Rn.as_matrix()[None]), Q[kk - 1, adr[3:]])[0]
        for i, nm in enumerate(jn):
            j = m.joint('robot_right_' + nm); col = int(j.qposadr[0]); lo, hi = j.range; tgt = rq[7 + i]
            if not nm.endswith('_AA'): tgt = lo + g['relax'] * (tgt - lo)
            Q[:, col] = np.clip((1 - w) * Q[:, col] + w * tgt, lo, hi)
        return Q
    def lilc(self, Q, Qx, c):
        """teacher_ilc_b.py left-hand part vs the c18 cup in physics (EPD tracking reference): left wrist rotated about the achieved cup by
        -lgr * (axis tilt error) and shifted by -lgp * (position error); errors smoothed, clamped to 4 cm / 25 deg, window [l0, l1]."""
        m = self.m; n = self.n; t = self.t; ca = int(m.joint('blue_cup_free').qposadr[0])
        if not hasattr(self, 'qt'): self.qt = TR.load_reference(EPD / 'tracking_reference.npz', m)['qpos']
        pc = Qx[:n, ca:ca + 3]; Rc = R.from_quat(Qx[:n, ca + 3:ca + 7], scalar_first=True); Rt = R.from_quat(self.qt[:n, ca + 3:ca + 7], scalar_first=True)
        u = Rc.apply(T.CUPAX); v = Rt.apply(T.CUPAX); cax = np.cross(v, u); sn = np.linalg.norm(cax, axis=1, keepdims=True); ang = np.arctan2(sn[:, 0], (u * v).sum(1))
        erc = np.where(sn > 1e-9, cax / np.maximum(sn, 1e-9), 0) * np.minimum(ang, np.radians(25))[:, None]; epc = pc - self.qt[:n, ca:ca + 3]
        epc = epc * np.minimum(1, 0.04 / np.maximum(np.linalg.norm(epc, axis=1, keepdims=True), 1e-9))
        erc = gaussian_filter1d(erc, c['lsig'] * self.fps, axis=0, mode='nearest'); epc = gaussian_filter1d(epc, c['lsig'] * self.fps, axis=0, mode='nearest')
        gl = sm((t - c['l0']) / 0.3) * (1 - sm((t - (c['l1'] - 0.3)) / 0.3)); a = self.adl; Q = Q.copy()
        for k in range(1, n):
            if gl[k] == 0: continue
            C = R.from_rotvec(-c['lgr'] * gl[k] * erc[k]); p0 = pc[k]
            Q[k, a[:3]] = p0 + C.apply(Q[k, a[:3]] - p0) - c['lgp'] * gl[k] * epc[k]
            Q[k, a[3:]] = continuous_euler_xyz(R.from_matrix((C * R.from_euler('XYZ', Q[k, a[3:]])).as_matrix()[None]), Q[k - 1, a[3:]])[0]
        return Q
    def plan(self, v):
        """commands + c['lilc'] left ILC iterations (each one rollout). Returns (Q, Qx of Q)."""
        c = dict(DEF); c.update(v); Q = self.commands(v); Qx = self.rollout(Q)
        for _ in range(int(c['lilc'])): Q = self.lilc(Q, Qx, c); Qx = self.rollout(Q)
        return Q, Qx
    def rollout(self, Q):
        d = mujoco.MjData(self.m); mujoco.mj_resetDataKeyframe(self.m, d, 0)
        return np.asarray(servo_rollout(self.m, Q, self.fps, qpos_start=d.qpos.copy(), record_substeps=False)['qpos'])
W = None; PERT = []
def init(base, scene, levels=(), npert=0):
    global W, PERT; W = Base(base, scene); PERT = [(lv, Z) for lv in levels for Z in noise_draws(W.m, W.n, lv, npert)]
def run(v):
    try:
        Q, Qx = W.plan(v); res, txt = T.table(W.m, Qx, json.dumps(v)); out = dict(v=v, res=res, txt=txt)
        if PERT:
            pr = [dict(level=lv, **drop_metrics(W.m, W.rollout(Q + Z))) for lv, Z in PERT]
            out['pert'] = dict(n=len(pr), pass_rate=float(np.mean([p['passed'] for p in pr])), spoon_drop_rate=float(np.mean([p['spoon_drop'] for p in pr])),
                               cup_lost_rate=float(np.mean([p['cup_lost'] for p in pr])), r_fingers=float(np.mean([p['r_fingers'] for p in pr])),
                               spoon_med=[float(np.median([p['summary']['wooden_spoon'][0] for p in pr])), float(np.median([p['summary']['wooden_spoon'][1] for p in pr]))],
                               runs=[dict(level=p['level'], passed=p['passed'], spoon_drop=p['spoon_drop'], cup_lost=p['cup_lost'], summary=p['summary']) for p in pr])
        return out
    except Exception as ex: return dict(v=v, error=repr(ex)[:300])
def short(res):
    rows = res['rows']; hold = [r for r in rows if 2.5 <= r['t0'] < 14.5]
    return dict(summary=res['summary'], r_fingers_hold=round(float(np.mean([r['r_fingers_mean'] for r in hold])), 2), r_min_win=round(min(r['r_fingers_mean'] for r in hold), 1),
                l_fingers_hold=round(float(np.mean([r['l_fingers_mean'] for r in hold])), 2), l_phi_med=round(float(np.median([r['l_phi'] for r in hold])), 0),
                spoon_max_win=[round(max(r['spoon'][0] for r in rows), 2), round(max(r['spoon'][1] for r in rows), 1)], cup_max_win=[round(max(r['cup'][0] for r in rows), 2), round(max(r['cup'][1] for r in rows), 1)])
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['grid', 'make']); ap.add_argument('base'); ap.add_argument('out'); ap.add_argument('--scene')
    ap.add_argument('--variants', help='grid: JSON list of dicts (keys of DEF) or @file'); ap.add_argument('--workers', type=int, default=12); ap.add_argument('--v', default='{}', help='make: JSON dict (keys of DEF)')
    ap.add_argument('--pert', default='', help='noise levels (rad|m on all robot command columns, team fcem_r model), e.g. 0.0005,0.001'); ap.add_argument('--npert', type=int, default=4)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); lv = [float(x) for x in a.pert.split(',') if x]
    if a.mode == 'grid':
        vs = json.loads(Path(a.variants[1:]).read_text() if a.variants.startswith('@') else a.variants); t0 = time.monotonic(); allr = []
        with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context('spawn'), initializer=init, initargs=(a.base, a.scene, lv, a.npert)) as pool:
            for r in pool.map(run, vs):
                if 'error' in r: print('ERR', json.dumps(r), flush=True); continue
                s = short(r['res']); s['pert'] = {k: v for k, v in r.get('pert', {}).items() if k != 'runs'}; allr.append(dict(v=r['v'], **s, pert_runs=r.get('pert', {}).get('runs')))
                print('VAR', json.dumps(r['v']), json.dumps(s), flush=True); print(r['txt'], flush=True)
        (out / 'grid.json').write_text(json.dumps(dict(base=a.base, results=allr, elapsed_s=time.monotonic() - t0), indent=1)); return
    v = json.loads(a.v); b = Base(a.base, a.scene); Q, _ = b.plan(v)
    refp = TR.save_reference(out / 'reference.npz', b.m, Q, b.ref['labels'], b.fps, b.ref['segment'], b.ref['segments'], source=f'lside33 {json.dumps(v)} from {a.base}', arms_coupled=False, floating_runtime=True)
    ex = execute_reference(b.scene, refp, out / 'execution'); Qx = np.load(ex['trajectory'])['qpos']; res, txt = T.table(b.m, Qx, f'lside33 make {json.dumps(v)}'); print(txt, flush=True)
    r0 = json.loads((b.base / 'result.json').read_text()); r0.update(execution=ex, lside33=dict(base=a.base, v=v, defaults=DEF, script=__file__), t33=dict(summary=res['summary'], short=short(res)))
    (out / 'result.json').write_text(json.dumps(r0, indent=1, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))); (out / 't33table.json').write_text(json.dumps(res, indent=1)); print('MADE', out, json.dumps(short(res)), flush=True)
if __name__ == '__main__': main()
