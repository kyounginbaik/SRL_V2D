"""Stream P ep33 phase 6: right-hand re-plan of a compose-world actor toward the compose spoon track (shared by the CEM search and
the final checkpoint writer). Base = an actor already left-refitted to the compose cup (C/t3_*), with its new-world rollout
(pickup works) and the old-world rollout of the same base weights (for the held-spoon estimate after the grasp frame f_g:
A_est(t) = Delta A_old(t), Delta = A_new(f_g) A_old(f_g)^-1).
Per substep the desired spoon correction is Delta(t) = A_est(t)^-1 N(t) (body frame; N = compose spoon in our setup frame):
 roll (body x) * g_roll on [t_on, t_off] (smoothstep 0.3 s), continuous-unwrapped, plus 360 deg * w(t) with w ramping 0->1 on
 [t_w0, t_w1] (the positive-branch trick: same orientations, avoids the wrist XYZ gimbal), kept to the end;
 tilt (body y, z) * g_tilt; position (world) * g_pos; plus a constant wrist offset (ox, oy, oz cm) on [to0, to1];
 pre-grasp: the fraction k_td of the resting-spoon offset T_d (new vs old rest) applied to the wrist before f_g - 6.
The wrist command is moved by the resulting spoon correction about the estimated spoon origin; feedback divided out; projected
onto the actor's phase basis."""
import os
import json, numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R, Slerp
import sys
B = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017'); sys.path[:0] = [str(B / 'code'), '/data1/SRL_V2D/third_party/DexAgent']
from wrist_reference import continuous_euler_xyz
from learned_feedback_policy import features
P = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamP_ep33'); OEP = P / os.environ.get('CW_OUT', 'staging_C') / 'episode_compose/episode_000033'
sm = lambda u: np.clip(u, 0, 1) ** 3 * (10 - 15 * np.clip(u, 0, 1) + 6 * np.clip(u, 0, 1) ** 2)
def M(p, q): return np.vstack([np.c_[R.from_quat(q[[1, 2, 3, 0]]).as_matrix(), p], [0, 0, 0, 1]])
class Plan:
    def __init__(self, base_dir, old_rollout, fg=62):
        base = Path(base_dir); ev = json.loads((base / 'evaluation/result.json').read_text()); self.scene = ev['execution']['scene']; self.base = base
        with np.load(base / 'training/policy.npz', allow_pickle=False) as z: self.d = {k: z[k] for k in z.files}
        tn = np.load(ev['execution']['trajectory']); to = np.load(old_rollout); n = self.n = 348; self.ft = ft = np.arange(n) / 20.0; self.fg = fg
        i = [str(x) for x in tn['object_names']].index('wooden_spoon'); io = [str(x) for x in to['object_names']].index('wooden_spoon')
        An = np.array([M(tn['object_positions'][f, i], tn['object_quaternions_wxyz'][f, i]) for f in range(n)]); Ao = np.array([M(to['object_positions'][f, io], to['object_quaternions_wxyz'][f, io]) for f in range(n)])
        D = An[fg] @ np.linalg.inv(Ao[fg]); A = An.copy(); A[fg + 1:] = np.einsum('ij,njk->nik', D, Ao[fg + 1:]); self.A = A
        self.Td = An[34] @ np.linalg.inv(Ao[34]); self.c_rest = Ao[34][:3, 3]     # resting spoon: new world vs old world (both before the touch)
        man = json.loads((OEP / 'input_manifest.json').read_text()); self.N = np.asarray(man['setup_from_source']) @ np.load(Path(man['source']) / 'object_poses.npz')['world_from_object'][:, 1]
        self.st = tn['substep_time_s']; self.sc = tn['substep_ctrl']
        rel = np.einsum('nji,njk->nik', A[:, :3, :3], self.N[:, :3, :3]); self.rv = R.from_matrix(rel).as_rotvec()
        roll = np.unwrap(self.rv[:, 0]); self.roll = roll; self.dpos = self.N[:, :3, 3] - A[:, :3, 3]
        na = [str(x) for x in self.d['actuator_names']]; pre = 'robot_right_right_'
        self.ids = [na.index(pre + k + '_position') for k in ('pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z')]
        self.Phi = features(self.st / float(self.d['duration_s']), self.d['centers'], float(self.d['width']))
        self.Ast_p = np.stack([np.interp(self.st, ft, A[:, k, 3]) for k in range(3)], 1); self.Ast_R = Slerp(ft, R.from_matrix(A[:, :3, :3]))(np.clip(self.st, 0, ft[-1]))
    def build(self, p):
        st, ft, d = self.st, self.ft, self.d
        g = sm((ft - p['t_on']) / 0.3) * (1 - sm((ft - p['t_off'] + 0.3) / 0.3))
        w = sm((ft - p['t_w0']) / max(p['t_w1'] - p['t_w0'], 0.05)) if p.get('wturn', 1) else np.zeros_like(ft)
        roll = p['g_roll'] * g * self.roll + np.radians(360.0) * w * (p.get('wturn', 1))
        tilt = p['g_tilt'] * g[:, None] * self.rv[:, 1:]
        bodyrv = np.c_[roll, tilt]; Rb = R.from_rotvec(np.stack([np.interp(st, ft, bodyrv[:, k]) for k in range(3)], 1))
        dpos = p['g_pos'] * g[:, None] * self.dpos; dps = np.stack([np.interp(st, ft, dpos[:, k]) for k in range(3)], 1)
        og = sm((st - p['to0']) / 0.25) * (1 - sm((st - (max(p['to1'], p['to0'] + 0.5) - 0.25)) / 0.25)); dps = dps + og[:, None] * np.array([p['ox'], p['oy'], p['oz']]) / 100
        # pre-grasp T_d fraction (fades out over the 0.3 s after the touch so the grasp is made at the shifted place and kept)
        k = p.get('k_td', 0.0); tdr = R.from_matrix(self.Td[:3, :3]).as_rotvec(); tdt = self.Td[:3, 3]
        C = self.Ast_R * Rb * self.Ast_R.inv()                      # world rotation about the estimated spoon
        p0 = self.sc[:, self.ids[:3]]; ang0 = self.sc[:, self.ids[3:]]; Rw = R.from_euler('XYZ', ang0)
        p1 = self.Ast_p + C.apply(p0 - self.Ast_p) + dps
        R1 = C * Rw
        if k:   # fractional rigid T_d about the old resting spoon origin c: x -> c + R_k (x - c) + kk (T_d(c) - c), kk = k * pre(t)
            kk = k * (1 - sm((st - self.fg / 20.0 + 0.6) / 0.3)); c = self.c_rest; shift = self.Td[:3, :3] @ c + self.Td[:3, 3] - c
            Ck = R.from_rotvec(kk[:, None] * tdr); p1 = c + Ck.apply(p1 - c) + kk[:, None] * shift; R1 = Ck * R1
        ang1 = ang0.copy()
        for kk in range(len(st)): ang1[kk] = continuous_euler_xyz(R.from_matrix(R1[kk].as_matrix()[None]), ang1[kk - 1] + (ang0[kk] - ang0[kk - 1]) if kk else ang0[kk])[0]
        D = np.concatenate([p1 - p0, ang1 - ang0], 1)
        selfg = np.array([d['feedback_weights'][j, j] / d['state_scale'][j] * d['action_scale'][j] for j in self.ids]); D = D * (1 - selfg)
        target = D / d['action_scale'][self.ids]; Phi = self.Phi
        Dw = np.linalg.solve(Phi.T @ Phi + 1e-3 * np.eye(Phi.shape[1]), Phi.T @ target); W = d['phase_weights'].copy(); W[:, self.ids] += Dw
        spd = float(np.abs(np.diff(ang1, axis=0) / np.diff(st)[:, None]).max())
        return W, dict(max_rot_joint_speed=spd, proj_err=float(np.abs(Phi @ Dw * d['action_scale'][self.ids] - D).max()))
