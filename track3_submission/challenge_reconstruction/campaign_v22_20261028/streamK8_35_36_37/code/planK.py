"""Stream K8 (v23) eps 35/36/37: COPY of streamC_ep3536/code/planC.py; only change: episode dir per EP (35/36 = the C7 target_final_v5 worlds, 37 = streamK8 world built by build_world.py). planC docstring: Stream C7 (v22) eps 35/36: COPY of campaign_v16 streamW_C36/code/plan36.py generalised to both episodes (EP from env STREAMC_EP),
reference = target_final_v5 retargeted world (streamC_ep3536/episode/episode_X, build_world.py). Roll term generalised: phi_r is applied
with sign rsign (C_roll = Rx(-rsign*phi_r)), so it can cancel (rsign=+1) or RE-TIME (rsign=-1: counter the base actor's own back-roll
at rc0-rc1 and roll back later at rc2-rc3) the base actor's half-turn. Added: up to 3 smooth roll PULSES about the executed spoon's long
(body x) axis, pulse k = amplitude pk_a deg, smoothstep rise pk_s0..pk_s1, fall pk_e0..pk_e1 (added to the roll term), to follow the
target's in-hand roll excursions (ep35 f70-86, f136-168, f276-336) without following its frame-level rotation noise.
FIX vs plan36: the wrist rot_x/y/z hinges sit under a fixed CHART rotation A (body_quat of robot_right_right_base_roll; identity in
ep36, a virtual wrist chart in ep35), R_palm = A R_XYZ(q). A world correction C therefore maps to the chart as A^-1 C A (plan36
assumed A = I, which turned every ep35 rotation correction into a rotation about the wrong axis). Original docstring follows.
Stream W ep36 (v21C): shared plan builder that makes a base actor's held spoon follow the compose-retargeted reference
(episode_compose). Used by cem36.py for the search AND for writing the final checkpoint (same function of the parameters).
Built on streamP_ep33/code/rollplan.py (fold the re-plan into the actor's phase weights; diagonal feedback divided out).

Frame-level target (setup frame): R_tgt(f) = R_orig(f) @ clip(R_orig(f)^-1 R_comp(f), clip deg)  [R_orig = our ORIGINAL (unrolled)
track; clip limits how much of the compose roll excursions the robot tries to follow], p_tgt = compose position.
Base execution: R_exec, p_exec (executed spoon). Optional roll cancel: the base actor rolls the spoon +180 deg about its long
axis (corrected-track roll, executed ramps ~12.15-12.85 s up / 14.55-15.0 s down); cancel = R_exec @ Rx(-phi_r(t)) with a
smoothstep profile [rc0, rc1] / [rc2, rc3] (the compose track has no such roll). Correction about the EXECUTED spoon origin:
C(t) = exp(gr * w(t) * smooth(cap(log(R_tgt(t+lead) Ru(t)^-1), cmax), sig)) @ C_roll(t); dp = gp * w * smooth(p_tgt - p_exec).
w(t): smoothstep 0.3 s in at t_on, out at t_off (held window). Wrist command: p1 = ps + C (p0 - ps) + dp, R1 = C Rw."""
import json, numpy as np, sys
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R, Slerp
B = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017'); sys.path[:0] = [str(B / 'code'), '/data1/SRL_V2D/third_party/DexAgent']
from wrist_reference import continuous_euler_xyz
from learned_feedback_policy import features
import os
EP = int(os.environ.get('STREAMC_EP', '36')); NAME = f'episode_{EP:06d}'
W = Path(__file__).resolve().parent.parent
WC = {35: W.parent / 'streamC_ep3536', 36: W.parent / 'streamC_ep3536', 37: W}[EP]
OEP = WC / 'episode' / NAME; ORIG = Path('/data1/SRL_V2D/challenge_reconstruction/final_recon') / NAME / 'object_poses.npz'
REC = WC / 'source' / NAME / 'target_retarget.json'
sm = lambda u: np.clip(u, 0, 1) ** 3 * (10 - 15 * np.clip(u, 0, 1) + 6 * np.clip(u, 0, 1) ** 2)
DEFAULT = dict(gr=1.0, gp=1.0, sig=0.15, clip=180.0, cmax=70.0, t_on=2.3, t_off=18.3, lead=0.0, cancel=0.0, rsign=1.0, rc0=12.15, rc1=12.85, rc2=14.55, rc3=15.0, squeeze=0.0, **{f'p{k}_{x}': 0.0 for k in (1, 2, 3) for x in ('a', 's0', 's1', 'e0', 'e1')})

def reference_targets():
    man = json.loads((OEP / 'input_manifest.json').read_text()); S = np.asarray(man['setup_from_source'])
    z = np.load(Path(man['source']) / 'object_poses.npz'); names = [str(x) for x in z['names']]; i = names.index('wooden_spoon')
    Tc = S @ z['world_from_object'][:, i]; To = S @ np.load(ORIG)['world_from_object'][:, i]; return Tc, To

class Plan:
    def __init__(self, base_dir):
        base = Path(base_dir); ev = json.loads((base / 'evaluation/result.json').read_text()); self.scene = ev['execution']['scene']; self.base = base
        with np.load(base / 'training/policy.npz', allow_pickle=False) as z: self.d = {k: z[k] for k in z.files}
        tr = np.load(ev['execution']['trajectory']); si = [str(x) for x in tr['object_names']].index('wooden_spoon'); self.n = n = int(round(float(self.d['frames']))); self.fps = 20.0
        self.ft = np.arange(n) / 20.0; self.st = tr['substep_time_s']; self.sc = tr['substep_ctrl']
        self.pe = tr['object_positions'][:n, si]; self.Re = R.from_quat(tr['object_quaternions_wxyz'][:n, si][:, [1, 2, 3, 0]])
        self.ps = np.stack([np.interp(self.st, self.ft, self.pe[:, i]) for i in range(3)], 1)
        self.Rs = Slerp(self.ft, self.Re)(np.clip(self.st, self.ft[0], self.ft[-1]))
        Tc, To = reference_targets(); self.Tc, self.To = Tc, To
        na = [str(x) for x in self.d['actuator_names']]; pre = 'robot_right_right_'
        self.ids = [na.index(pre + k + '_position') for k in ('pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z')]
        self.pinch = [na.index(pre + j + '_position') for j in ('thumb_CMC_FE', 'thumb_MCP_FE', 'thumb_IP', 'index_PIP', 'index_DIP')]
        import mujoco as _mj; _m = _mj.MjModel.from_xml_path(self.scene); q = _m.body_quat[_m.body('robot_right_right_base_roll').id]
        self.A = R.from_quat([q[1], q[2], q[3], q[0]])
        self.Phi = features(self.st / float(self.d['duration_s']), self.d['centers'], float(self.d['width']))
    def phi_r(self, t, p):
        """total roll (deg, applied as Rx(-phi)): rsign * base-roll cancel/re-time + pulses"""
        out = np.zeros_like(t, dtype=float)
        if p.get('cancel', 0) >= 0.5:
            out += p.get('rsign', 1.0) * 180.0 * (sm((t - p['rc0']) / max(p['rc1'] - p['rc0'], 0.05)) - sm((t - p['rc2']) / max(p['rc3'] - p['rc2'], 0.05)))
        for k in (1, 2, 3):
            a = p.get(f'p{k}_a', 0.0)
            if abs(a) > 1e-9:
                out += a * (sm((t - p[f'p{k}_s0']) / max(p[f'p{k}_s1'] - p[f'p{k}_s0'], 0.05)) - sm((t - p[f'p{k}_e0']) / max(p[f'p{k}_e1'] - p[f'p{k}_e0'], 0.05)))
        return out
    def targets(self, p):
        ft = self.ft; tq = np.clip(ft + p.get('lead', 0.0), 0, ft[-1])
        Ro = R.from_matrix(self.To[:, :3, :3]); Rc = R.from_matrix(self.Tc[:, :3, :3])
        D = (Ro.inv() * Rc).as_rotvec(); m = np.linalg.norm(D, axis=1); D *= np.minimum(1, np.radians(p['clip']) / np.maximum(m, 1e-9))[:, None]
        Rt = Ro * R.from_rotvec(D); Rt = Slerp(ft, Rt)(tq); pt = np.stack([np.interp(tq, ft, self.Tc[:, i, 3]) for i in range(3)], 1); return Rt, pt
    def corrections(self, p):
        """frame-level smoothed small correction rotvec v (world) and translation dp, plus the window weight."""
        ft = self.ft; Rt, pt = self.targets(p)
        Ru = self.Re * R.from_euler('x', -self.phi_r(ft, p)[:, None], degrees=True)
        v = (Rt * Ru.inv()).as_rotvec(); m = np.linalg.norm(v, axis=1); v *= np.minimum(1, np.radians(p['cmax']) / np.maximum(m, 1e-9))[:, None]
        dp = pt - self.pe; s = p['sig'] * 20.0
        if s > 0.05: v = gaussian_filter1d(v, s, axis=0, mode='nearest'); dp = gaussian_filter1d(dp, s, axis=0, mode='nearest')
        w = sm((ft - p['t_on']) / 0.3) * (1 - sm((ft - (p['t_off'] - 0.3)) / 0.3))
        return v * (p['gr'] * w)[:, None], dp * (p['gp'] * w)[:, None], w
    def build(self, p):
        q = dict(DEFAULT); q.update(p); p = q
        d = self.d; st, sc = self.st, self.sc; ids = self.ids
        v, dp, w = self.corrections(p)
        vs = np.stack([np.interp(st, self.ft, v[:, i]) for i in range(3)], 1); dps = np.stack([np.interp(st, self.ft, dp[:, i]) for i in range(3)], 1)
        phis = self.phi_r(st, p)
        act = (np.linalg.norm(vs, axis=1) > 1e-9) | (np.abs(phis) > 1e-9) | (np.linalg.norm(dps, axis=1) > 1e-12)
        p0 = sc[:, ids[:3]]; ang0 = sc[:, ids[3:]]; Rw = R.from_euler('XYZ', ang0); p1 = p0.copy(); ang1 = ang0.copy()
        k = np.flatnonzero(act)
        Croll = self.Rs[k] * R.from_euler('x', -phis[k][:, None], degrees=True) * self.Rs[k].inv()
        C = R.from_rotvec(vs[k]) * Croll; Cc = self.A.inv() * C * self.A  # world correction C (positions); chart conjugate Cc (wrist angles)
        p1[k] = self.ps[k] + C.apply(p0[k] - self.ps[k]) + dps[k]; R1 = Cc * Rw[k]; prev = None
        for j, kk in enumerate(k):
            anchor = ang0[kk] if prev is None or kk - prev > 1 else ang1[prev] + (ang0[kk] - ang0[prev])
            ang1[kk] = continuous_euler_xyz(R.from_matrix(R1[j].as_matrix()[None]), anchor)[0]; prev = kk
        delta = np.concatenate([p1 - p0, ang1 - ang0], 1); allids = list(ids); cols = [delta]
        if abs(p.get('squeeze', 0)) > 1e-6:
            g = np.interp(st, self.ft, w)
            for j in self.pinch: allids.append(j); cols.append((np.clip(sc[:, j] + p['squeeze'] * g, d['ctrl_lo'][j], d['ctrl_hi'][j]) - sc[:, j])[:, None])
        Dm = np.concatenate(cols, 1)
        selfg = np.array([d['feedback_weights'][j, j] / d['state_scale'][j] * d['action_scale'][j] for j in allids]); Dm = Dm * (1 - selfg)
        use = (np.abs(Dm) > 1e-12).any(1)
        if not use.any(): return d['phase_weights'].copy(), dict(max_rot_change=0.0, proj_err=0.0)
        active = np.nonzero(self.Phi[use].max(0) > 1e-6)[0]
        A = self.Phi[:, active]; target = Dm / d['action_scale'][allids]
        Dw = np.linalg.solve(A.T @ A + 1e-3 * np.eye(len(active)), A.T @ target)
        Wn = d['phase_weights'].copy(); Wn[np.ix_(active, allids)] += Dw
        return Wn, dict(max_rot_change=float(np.abs(delta[:, 3:]).max()), proj_err=float(np.abs(A @ Dw * d['action_scale'][allids] - Dm).max()))
