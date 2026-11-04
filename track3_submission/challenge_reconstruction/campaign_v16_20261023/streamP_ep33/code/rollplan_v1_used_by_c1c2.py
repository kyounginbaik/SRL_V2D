"""Stream P ep33 phase 5b: shared plan builder for the grasp-aware right-hand roll re-plan (used by roll_cem.py for the
search AND for writing the final checkpoint, so the searched and the staged actor are the same function of the parameters).
Parameters (dict): t_s, t_r [s] (robot roll ramps 0 -> 360 + phi_ref(t_r) over [t_s, t_r], PCHIP), lead [s] (after t_r the robot
roll follows 360 + phi_ref(t + lead), i.e. the corrected reference continued on the positive branch, which reaches 360 deg =
the original orientation at f165 and is then held: same physical orientation, wrist joints wound; the positive branch avoids the
XYZ wrist gimbal that the negative branch crosses), ox, oy, oz [cm] world translation of the right wrist over [to0, to1]
(0.25 s smoothstep ramps), squeeze [rad] closing bias on the thumb-index pinch over [t_s - 0.3, 8.8] s.
The wrist command is rotated about the EXECUTED spoon long axis of the base rollout; the command change (divided by the actor's
diagonal feedback, -0.1 per joint) is least-squares projected onto the actor's own radial phase basis (phase_weights)."""
import json, numpy as np
from pathlib import Path
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation as R, Slerp
import sys
B = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017'); sys.path[:0] = [str(B / 'code'), '/data1/SRL_V2D/third_party/DexAgent']
from wrist_reference import continuous_euler_xyz
from learned_feedback_policy import features
P = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamP_ep33')
OEP = P / 'staging/episode_corrected/episode_000033'; REC = P / 'staging/source_corrected/episode_000033/roll_correction.json'
sm = lambda u: np.clip(u, 0, 1) ** 3 * (10 - 15 * np.clip(u, 0, 1) + 6 * np.clip(u, 0, 1) ** 2)

class Plan:
    def __init__(self, base_dir):
        base = Path(base_dir); ev = json.loads((base / 'evaluation/result.json').read_text()); self.scene = ev['execution']['scene']; self.base = base
        with np.load(base / 'training/policy.npz', allow_pickle=False) as z: self.d = {k: z[k] for k in z.files}
        tr = np.load(ev['execution']['trajectory']); si = [str(x) for x in tr['object_names']].index('wooden_spoon'); self.n = n = 348; self.fps = 20.0
        self.ft = np.arange(n) / 20.0; self.st = tr['substep_time_s']; self.sc = tr['substep_ctrl']
        self.ps = np.stack([np.interp(self.st, self.ft, tr['object_positions'][:n, si, i]) for i in range(3)], 1)
        self.Rs = Slerp(self.ft, R.from_quat(tr['object_quaternions_wxyz'][:n, si][:, [1, 2, 3, 0]]))(np.clip(self.st, self.ft[0], self.ft[-1]))
        rec = json.loads(REC.read_text()); self.phi_ref = np.zeros(n)
        for k, v in rec['phi_deg_per_frame'].items(): self.phi_ref[int(k)] = v
        na = [str(x) for x in self.d['actuator_names']]; pre = 'robot_right_right_'
        self.ids = [na.index(pre + k + '_position') for k in ('pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z')]
        self.pinch = [na.index(pre + j + '_position') for j in ('thumb_CMC_FE', 'thumb_MCP_FE', 'thumb_IP', 'index_PIP', 'index_DIP')]
        self.Phi = features(self.st / float(self.d['duration_s']), self.d['centers'], float(self.d['width']))
        self.f0 = int(np.searchsorted(self.ft, 5.0)); self.f_end = int(round(165))
    def roll_profile(self, p):
        st = self.st; phi = np.zeros(len(st)); ts, tr_ = p['t_s'], max(p['t_r'], p['t_s'] + 0.15)
        cont = lambda t: 360.0 + np.interp(t + p['lead'], self.ft, self.phi_ref)
        A = float(cont(tr_)); m1 = (st >= ts) & (st <= tr_)
        phi[m1] = PchipInterpolator([ts, (ts + tr_) / 2, tr_], [0.0, A / 2, A])(st[m1])
        m2 = st > tr_; c = cont(st[m2]); c[(st[m2] + p['lead']) >= self.f_end / 20.0] = 360.0; phi[m2] = c
        return phi
    def build(self, p):
        d = self.d; st, sc = self.st, self.sc; ids = self.ids
        phi = self.roll_profile(p); act = phi != 0
        p0 = sc[:, ids[:3]]; ang0 = sc[:, ids[3:]]; Rw = R.from_euler('XYZ', ang0); p1 = p0.copy(); ang1 = ang0.copy()
        k_act = np.flatnonzero(act)
        C = self.Rs[k_act] * R.from_euler('x', phi[k_act][:, None], degrees=True) * self.Rs[k_act].inv()
        p1[k_act] = self.ps[k_act] + C.apply(p0[k_act] - self.ps[k_act]); R1 = C * Rw[k_act]; prev = None
        for j, k in enumerate(k_act):
            anchor = ang0[k] if prev is None or k - prev > 1 else ang1[prev] + (ang0[k] - ang0[prev])
            ang1[k] = continuous_euler_xyz(R.from_matrix(R1[j].as_matrix()[None]), anchor)[0]; prev = k
        delta = np.concatenate([p1 - p0, ang1 - ang0], 1)
        og = sm((st - p['to0']) / 0.25) * (1 - sm((st - (max(p['to1'], p['to0'] + 0.5) - 0.25)) / 0.25))
        delta[:, :3] += og[:, None] * np.array([p['ox'], p['oy'], p['oz']]) / 100
        allids = list(ids); cols = [delta]
        if abs(p.get('squeeze', 0)) > 1e-6:
            g = sm((st - (p['t_s'] - 0.3)) / 0.3) * (1 - sm((st - 8.5) / 0.3))
            for j in self.pinch: allids.append(j); cols.append((np.clip(sc[:, j] + p['squeeze'] * g, d['ctrl_lo'][j], d['ctrl_hi'][j]) - sc[:, j])[:, None])
        D = np.concatenate(cols, 1)
        selfg = np.array([d['feedback_weights'][j, j] / d['state_scale'][j] * d['action_scale'][j] for j in allids]); D = D * (1 - selfg)
        use = (np.abs(D) > 1e-12).any(1); active = np.nonzero(self.Phi[use].max(0) > 1e-6)[0]
        A = self.Phi[:, active]; target = D / d['action_scale'][allids]
        Dw = np.linalg.solve(A.T @ A + 1e-3 * np.eye(len(active)), A.T @ target)
        W = d['phase_weights'].copy(); W[np.ix_(active, allids)] += Dw
        return W, dict(max_rot_change=float(np.abs(delta[:, 3:]).max()), proj_err=float(np.abs(A @ Dw * d['action_scale'][allids] - D).max()))
