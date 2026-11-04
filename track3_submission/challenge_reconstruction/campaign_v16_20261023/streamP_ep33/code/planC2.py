"""Stream P ep33 phase 6: two-hand compose-target re-plan of the v21 actor (built from its OLD-world rollout, applied in the compose
world staging_C), shared by the CEM search (cemC2.py) and the final checkpoint writer.
Left (cup): wrist corrected by C_L(t) = S_L*(t) A_cup(t)^-1 about the achieved cup, S_L* = blend(T_dc A_cup(t), N_cup(t), b_L(t)),
  T_dc = resting cup offset (port rollout in the new world vs old rollout, axis-aligning rotation), b_L = smoothstep(tl0, tl1) with
  gains (lp, lr); only the axis-aligning rotation (cup axially symmetric).
Right (spoon): the port pickup is kept (it works); after the grasp the held spoon is estimated as A_est = Dg A_old(t) (Dg from the
  port rollout at the grasp frame). Target position S_R*(t) = A_est + g_fc dcup(t) + g_sp (N_sp - A_est - dcup), dcup = N_cup - A_cup_old
  (keeps the spoon-in-cup geometry when the cup is carried along the new reference), rotation slerp(A_est, N_sp, g_rot) (shortest arc),
  all on [t_on, 17.3] (0.3 s ramps); plus a windowed offset (ox, oy, oz cm on [to0, to1]) and a thumb-index squeeze (sq rad on [t_on, 9 s]).
Command change: feedback divided out, least-squares projected onto the actor's phase basis."""
import os, json, numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.ndimage import gaussian_filter1d
import sys
B = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017'); sys.path[:0] = [str(B / 'code'), '/data1/SRL_V2D/third_party/DexAgent']
from wrist_reference import continuous_euler_xyz
from learned_feedback_policy import features
P = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamP_ep33'); CW = P / os.environ.get('CW_OUT', 'staging_C')
OEP = CW / 'episode_compose/episode_000033'; SC = CW / 'scene/scene_physics.xml'
CUPAX = np.array([-0.0237963577378219, -0.005046973670649213, 0.9997040869253169])
sm = lambda u: np.clip(u, 0, 1) ** 3 * (10 - 15 * np.clip(u, 0, 1) + 6 * np.clip(u, 0, 1) ** 2)
def M(p, q): return np.vstack([np.c_[R.from_quat(q[[1, 2, 3, 0]]).as_matrix(), p], [0, 0, 0, 1]])
def axis_rot(Ra, Rb):
    u, v = Ra @ CUPAX, Rb @ CUPAX; c = np.cross(u, v); s = np.linalg.norm(c); return R.from_rotvec(c / s * np.arctan2(s, np.dot(u, v)) if s > 1e-9 else np.zeros(3)).as_matrix()
class Plan:
    def __init__(self, base_dir, port_rollout, fg=62, rest=(12, 34)):
        import hashlib
        base = Path(base_dir); ev = json.loads((base / 'evaluation/result.json').read_text()); self.base = base; self.scene = str(SC)
        with np.load(base / 'training/policy.npz', allow_pickle=False) as z: self.d = {k: z[k] for k in z.files}
        self.d['scene_sha256'] = np.array(hashlib.sha256(SC.read_bytes()).hexdigest())
        to = np.load(ev['execution']['trajectory']); tp = np.load(port_rollout); n = self.n = 348; ft = self.ft = np.arange(n) / 20.0
        self.st = to['substep_time_s']; self.sc = to['substep_ctrl']; on = [str(x) for x in to['object_names']]; pn = [str(x) for x in tp['object_names']]
        Ao = {o: np.array([M(to['object_positions'][f, on.index(o)], to['object_quaternions_wxyz'][f, on.index(o)]) for f in range(n)]) for o in on}
        Ap = {o: np.array([M(tp['object_positions'][f, pn.index(o)], tp['object_quaternions_wxyz'][f, pn.index(o)]) for f in range(n)]) for o in pn}
        man = json.loads((OEP / 'input_manifest.json').read_text()); W = np.asarray(man['setup_from_source']) @ np.load(Path(man['source']) / 'object_poses.npz')['world_from_object']
        self.Ncup, self.Nsp = W[:, 0], W[:, 1]; self.Acup = Ao['blue_cup']
        O0, N0 = Ao['blue_cup'][rest[0]], Ap['blue_cup'][rest[0]]; self.Rdc = axis_rot(O0[:3, :3], N0[:3, :3]); self.O0c, self.N0c = O0[:3, 3], N0[:3, 3]
        Dg = Ap['wooden_spoon'][fg] @ np.linalg.inv(Ao['wooden_spoon'][fg]); A = Ao['wooden_spoon'].copy(); A[fg:] = np.einsum('ij,njk->nik', Dg, Ao['wooden_spoon'][fg:]); self.Asp = A
        self.dcup = gaussian_filter1d(self.Ncup[:, :3, 3] - Ao['blue_cup'][:, :3, 3], 3, axis=0)
        na = [str(x) for x in self.d['actuator_names']]
        self.idsL = [na.index(f'robot_left_left_{k}_position') for k in ('pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z')]
        self.idsR = [na.index(f'robot_right_right_{k}_position') for k in ('pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z')]
        self.pinch = [na.index(f'robot_right_right_{j}_position') for j in ('thumb_CMC_FE', 'thumb_MCP_FE', 'thumb_IP', 'index_PIP', 'index_DIP')]
        self.Phi = features(self.st / float(self.d['duration_s']), self.d['centers'], float(self.d['width'])); self.PtP = self.Phi.T @ self.Phi
    def _wrist(self, ids, A, tp, tR):
        st, ft = self.st, self.ft
        Cr = R.from_matrix(np.einsum('nij,nkj->nik', tR, A[:, :3, :3])); Crs = Slerp(ft, Cr)(np.clip(st, 0, ft[-1]))
        aps = np.stack([np.interp(st, ft, A[:, k, 3]) for k in range(3)], 1); tps = np.stack([np.interp(st, ft, tp[:, k]) for k in range(3)], 1)
        p0 = self.sc[:, ids[:3]]; ang0 = self.sc[:, ids[3:]]; p1 = tps + Crs.apply(p0 - aps); R1 = Crs * R.from_euler('XYZ', ang0); ang1 = ang0.copy()
        for k in range(len(st)): ang1[k] = continuous_euler_xyz(R.from_matrix(R1[k].as_matrix()[None]), ang1[k - 1] + (ang0[k] - ang0[k - 1]) if k else ang0[k])[0]
        return np.concatenate([p1 - p0, ang1 - ang0], 1), float(np.abs(np.diff(ang1, axis=0) / np.diff(st)[:, None]).max())
    def build(self, p):
        d, ft, st, n = self.d, self.ft, self.st, self.n
        bL = sm((ft - p['tl0']) / max(p['tl1'] - p['tl0'], 0.05))
        tpL, tRL = np.zeros((n, 3)), np.zeros((n, 3, 3))
        for f in range(n):
            p1 = self.N0c + self.Rdc @ (self.Acup[f, :3, 3] - self.O0c); R1 = self.Rdc @ self.Acup[f, :3, :3]; R2 = axis_rot(R1, self.Ncup[f, :3, :3]) @ R1
            tpL[f] = p1 + p['lp'] * bL[f] * (self.Ncup[f, :3, 3] - p1); tRL[f] = (R.from_matrix(R1) * R.from_rotvec(p['lr'] * bL[f] * (R.from_matrix(R1).inv() * R.from_matrix(R2)).as_rotvec())).as_matrix()
        DL, spdL = self._wrist(self.idsL, self.Acup, tpL, tRL)
        gR = sm((ft - p['t_on']) / 0.3) * (1 - sm((ft - 17.0) / 0.3)); A = self.Asp
        tpR = A[:, :3, 3] + gR[:, None] * (p['g_fc'] * self.dcup + p['g_sp'] * (self.Nsp[:, :3, 3] - A[:, :3, 3] - self.dcup))
        rel = R.from_matrix(np.einsum('nji,njk->nik', A[:, :3, :3], self.Nsp[:, :3, :3])).as_rotvec()
        tRR = (R.from_matrix(A[:, :3, :3]) * R.from_rotvec(p['g_rot'] * gR[:, None] * rel)).as_matrix()
        DR, spdR = self._wrist(self.idsR, A, tpR, tRR)
        og = sm((st - p['to0']) / 0.25) * (1 - sm((st - (max(p['to1'], p['to0'] + 0.5) - 0.25)) / 0.25)); DR[:, :3] += og[:, None] * np.array([p['ox'], p['oy'], p['oz']]) / 100
        ids = self.idsL + self.idsR; cols = [DL, DR]
        if abs(p.get('sq', 0)) > 1e-6:
            g = sm((st - p['t_on']) / 0.3) * (1 - sm((st - 8.7) / 0.3))
            for j in self.pinch: ids.append(j); cols.append((np.clip(self.sc[:, j] + p['sq'] * g, d['ctrl_lo'][j], d['ctrl_hi'][j]) - self.sc[:, j])[:, None])
        D = np.concatenate(cols, 1); D = D * (1 - np.array([d['feedback_weights'][j, j] / d['state_scale'][j] * d['action_scale'][j] for j in ids]))
        Dw = np.linalg.solve(self.PtP + 1e-3 * np.eye(self.PtP.shape[0]), self.Phi.T @ (D / d['action_scale'][ids])); Wn = d['phase_weights'].copy(); Wn[:, ids] += Dw
        return Wn, dict(max_rot_joint_speed=max(spdL, spdR), proj_err=float(np.abs(self.Phi @ Dw * d['action_scale'][ids] - D).max()))
