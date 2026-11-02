"""Mismatch-driven upper-level updates for UniRetarget (2026-10-28), copied into an arm by patch_mismatch_upper.py.

The original upper ('search') scores candidate references with one scalar cost and keeps the best. The modes here use the
per-frame physical mismatch itself to choose the edit (+bilevel.upper_update=...):

  none         no edit; the lower trains on the starting reference; the stop rule reads its own baseline replicas
  search       original: upper-SAC samples + ILC menu (unchanged code path)
  search_only  upper-SAC samples, no ILC menu (builder sets ilc_candidates=false; no code here)
  ilc          iterative learning control: every candidate is r - K(g) e over a gain grid (a line search along the mismatch)
  ensemble_gn  Gauss-Newton step on J = d(mismatch)/d(edit), ridge-regressed from the previous rounds' probe rollouts
  surrogate    gradient descent through an ensemble MLP model of the mismatch, trained on every round's rollouts
  ik_feedback  object-frame hand mismatch -> corrected IK targets (palm pose, contact fingertips) -> IK re-solve

Every mode fills the same C candidate slots (C x R replicas + NB baseline replicas), with the same cost and 2-SE gate.
Candidates from these modes are not upper-SAC samples: the editor zeroes their SAC actions and rewards (like ILC).
"""
import json
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

MODES = ('none', 'search', 'search_only', 'ilc', 'ensemble_gn', 'surrogate', 'surrogate2', 'ik_feedback')


def unit_quat(q):
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return np.where(n > 1e-8, q / np.maximum(n, 1e-8), np.array([0., 0., 0., 1.]))


def mismatch_features(object_pose, stride=2):
    """(Fm, W, O, 14) actual|target pose -> (W, K): object position (/3 cm) and rotation (/0.3 rad) error per frame,
    each world's frame-0 offset removed (as the ILC candidates and the MP-SR alignment do)."""
    OP = np.nan_to_num(np.asarray(object_pose, np.float64))
    Fm, W, O = OP.shape[:3]
    pos = OP[..., 0:3] - OP[..., 7:10]
    pos = pos - pos[:1]
    E = (Rotation.from_quat(unit_quat(OP[..., 3:7]).reshape(-1, 4))
         * Rotation.from_quat(unit_quat(OP[..., 10:14]).reshape(-1, 4)).inv()).as_quat().reshape(Fm, W, O, 4)
    E0 = np.repeat(E[:1], Fm, 0)
    rot = (Rotation.from_quat(E.reshape(-1, 4)) * Rotation.from_quat(E0.reshape(-1, 4)).inv()).as_rotvec().reshape(Fm, W, O, 3)
    f = np.concatenate([pos / .03, rot / .3], -1)[::stride]
    return np.clip(f, -10., 10.).transpose(1, 0, 2, 3).reshape(W, -1)


def stable_worlds(editor, min_count=16):
    sig = np.nan_to_num(np.asarray(editor._signed_object_offset, np.float64), nan=9., posinf=9., neginf=-9.)
    keep = np.linalg.norm(sig, axis=-1).max(axis=(0, 2)) < .3
    return keep if keep.sum() >= min_count else np.isfinite(editor._signed_object_offset).all(axis=(0, 2, 3))


def targets_for(editor, q):
    t = editor.baseline.joint_target.copy()
    t[:, editor.projector.dids] = q[:, editor.projector.qids]
    for qi, di in editor.projector.ball_joints:
        t[:, di:di + 3] = Rotation.from_quat(q[:, qi:qi + 4]).as_rotvec()
    return t


def action_scale(editor):
    """Per-channel scale so that a = +-1 reaches the channel's bound at the basis peak.

    floating_candidate rescales a whole channel trajectory to its bound (bounded_delta) whenever the raw edit exceeds it;
    raw upper actions in [-1, 1] are metres / radians, so any |a| above ~0.08 saturates and the map is scale-free.
    The regression modes need an edit that is linear in a: pre-scaling keeps bounded_delta inactive except on overflow."""
    if getattr(editor, '_mm_scale', None) is not None:
        return editor._mm_scale
    nch, nj = editor.basis.shape[1], editor.joint_basis.shape[0]
    s = np.ones(nj)
    try:
        p = editor.projector; names = list(p.layout.scalar_joints.names)
        trans = {qid for n, qid in zip(names, p.qids) if n.endswith(('wrist_x', 'wrist_y', 'wrist_z'))}
        rot = {st + i for st, _ in p.ball_joints for i in range(3)}
        tb = float(editor.options.get('wrist_translation_offset_m', .03)); rb = float(editor.options.get('wrist_rotation_offset_rad', .52)) / np.sqrt(3)
        fb = float(editor.options.get('finger_offset_rad', .52))
        peak_t = max(float(np.abs(editor.basis).max()), 1e-9)
        for c in range(nj):
            row = np.asarray(editor.joint_basis[c], np.float64); nz = set(np.flatnonzero(np.abs(row) > 1e-9))
            bound = tb if nz & trans else rb if nz & rot else fb
            s[c] = bound / (max(float(np.abs(row).max()), 1e-9) * peak_t)
    except AttributeError:   # unit-test editors without a robot layout
        pass
    editor._mm_scale = np.tile(s, nch)
    return editor._mm_scale


def realize(editor, a):
    """Upper action -> absolute reference (blend 1: the action alone defines the edit from the IK baseline)."""
    a = np.asarray(a, np.float64)
    s = action_scale(editor)
    a = np.r_[a[:len(s)] * s, a[len(s):]]
    blend = editor.blend
    editor.blend = 1.
    try:
        q = editor.candidate_joint_q(a.astype(np.float32))
    finally:
        editor.blend = blend
    return q, targets_for(editor, q)


def editable_mask(editor):
    nch, nj = editor.basis.shape[1], editor.joint_basis.shape[0]
    m = np.ones((nch, nj), bool)
    ch = getattr(editor, '_win_chunks', None)
    if ch is not None and len(ch) == nch:
        m[~np.asarray(ch, bool)] = False
    return m.reshape(-1)


def current_rollout(editor, learner):
    """Rollout of the current reference on every world (reuses the failure-window rollout of this round when present)."""
    if not getattr(editor, '_offset_cached', False):
        from flash_chord.retargeting.parallel_rollout import rollout_candidates
        W = editor.upper_worlds
        bq, bt = editor.binding.joint_q.copy(), editor.binding.joint_target.copy()
        rollout_candidates(editor, learner, np.repeat(bq[None], W, 0), np.repeat(bt[None], W, 0))
    editor._offset_cached = False
    return stable_worlds(editor)


# ----------------------------------------------------------------------------------------------- Jacobian / Gauss-Newton
def fit_jacobian(history, D, *, rounds=3, ridge=1e-2, rng=None):
    """J (K x D) with dM ~ dA J^T, ridge regression over the last `rounds` rounds (weight 0.5^age)."""
    A, M, w = [], [], []
    for age, h in enumerate(reversed(history[-rounds:])):
        A.append(h['dA']); M.append(h['dM']); w.append(np.full(len(h['dA']), .5 ** age))
    A, M, w = np.concatenate(A), np.concatenate(M), np.concatenate(w)
    if rng is not None:
        i = rng.integers(0, len(A), len(A)); A, M, w = A[i], M[i], w[i]
    G = (A * w[:, None]).T @ A
    lam = ridge * max(np.trace(G) / D, 1e-9)
    return np.linalg.solve(G + lam * np.eye(D), (A * w[:, None]).T @ M).T


def gauss_newton_step(J, m0, editable, mu):
    Je = J[:, editable]
    H = Je.T @ Je
    s = max(np.trace(H) / max(editable.sum(), 1), 1e-9)
    step = np.zeros(J.shape[1])
    step[editable] = -np.linalg.solve(H + mu * s * np.eye(len(H)), Je.T @ m0)
    return step


def probes(a0, editable, n, sigmas, rng):
    out = []
    for i in range(n):
        a = a0.copy()
        a[editable] += sigmas[i % len(sigmas)] * rng.standard_normal(int(editable.sum()))
        out.append(np.clip(a, -1., 1.))
    return out


# ----------------------------------------------------------------------------------------------- surrogate ensemble
class Surrogate:
    """Ensemble of MLPs g_k(a) with g(a_c) - g(a0) ~ m_c - m_0 (differences within one round cancel lower-policy drift)."""

    def __init__(self, D, K, members=5, hidden=256, seed=0):
        import jax
        import jax.numpy as jnp
        self.jax, self.jnp = jax, jnp
        self.D, self.K, self.members = D, K, members
        keys = jax.random.split(jax.random.PRNGKey(seed), members)
        self.params = [self._init(k, D, K, hidden) for k in keys]
        self.opt = [None] * members
        self._train_step = jax.jit(self._make_train_step())
        self._plan_step = jax.jit(self._make_plan_step())

    def _init(self, key, D, K, H):
        jax, jnp = self.jax, self.jnp
        k1, k2, k3 = jax.random.split(key, 3)
        s = lambda k, i, o, g=1.: jax.random.normal(k, (i, o)) * np.sqrt(1. / i) * g
        return dict(W1=s(k1, D, H), b1=jnp.zeros(H), W2=s(k2, H, H), b2=jnp.zeros(H), W3=s(k3, H, K, .1), b3=jnp.zeros(K))

    def f(self, p, a):
        jnp = self.jnp
        h = jnp.tanh(a @ p['W1'] + p['b1'])
        h = jnp.tanh(h @ p['W2'] + p['b2'])
        return h @ p['W3'] + p['b3']

    def _make_train_step(self):
        jax, jnp = self.jax, self.jnp

        def loss(p, A, A0, M, w):
            d = self.f(p, A) - self.f(p, A0) - M
            return jnp.sum(w[:, None] * d ** 2) / jnp.sum(w) / M.shape[1] + 1e-5 * sum(jnp.sum(x ** 2) for x in jax.tree_util.tree_leaves(p))

        def step(p, opt, A, A0, M, w, lr):
            g = jax.grad(loss)(p, A, A0, M, w)
            m, v, t = opt
            t = t + 1
            m = jax.tree_util.tree_map(lambda a, b: .9 * a + .1 * b, m, g)
            v = jax.tree_util.tree_map(lambda a, b: .999 * a + .001 * b * b, v, g)
            p = jax.tree_util.tree_map(lambda x, a, b: x - lr * (a / (1 - .9 ** t)) / (jnp.sqrt(b / (1 - .999 ** t)) + 1e-8), p, m, v)
            return p, (m, v, t), loss(p, A, A0, M, w)
        return step

    def fit(self, history, steps=300, decay=.7, lr=1e-3, seed=0):
        jnp = self.jnp
        rng = np.random.default_rng(seed)
        A = np.concatenate([h['A'] for h in history]); A0 = np.concatenate([np.repeat(h['a0'][None], len(h['A']), 0) for h in history])
        M = np.concatenate([h['dM'] for h in history])
        w = np.concatenate([np.full(len(h['A']), decay ** (len(history) - 1 - i)) for i, h in enumerate(history)])
        losses = []
        for k in range(self.members):
            i = rng.integers(0, len(A), len(A))   # bootstrap per member
            args = [jnp.asarray(x[i], jnp.float32) for x in (A, A0, M, w)]
            if self.opt[k] is None:
                z = self.jax.tree_util.tree_map(jnp.zeros_like, self.params[k]); self.opt[k] = (z, z, 0)
            for _ in range(steps):
                self.params[k], self.opt[k], l = self._train_step(self.params[k], self.opt[k], *args, lr)
            losses.append(float(l))
        return losses

    def _make_plan_step(self):
        jax, jnp = self.jax, self.jnp

        def objective(a, p, a0, m0, mask, beta):
            pred = m0 + self.f(p, a) - self.f(p, a0)
            return jnp.mean(pred ** 2) + beta * jnp.sum(mask * (a - a0) ** 2) / jnp.maximum(jnp.sum(mask), 1.)

        def step(a, opt, p, a0, m0, mask, beta, lr):
            g = jax.grad(objective)(a, p, a0, m0, mask, beta) * mask
            m, v, t = opt
            t = t + 1
            m = .9 * m + .1 * g; v = .999 * v + .001 * g * g
            a = jnp.clip(a - lr * (m / (1 - .9 ** t)) / (jnp.sqrt(v / (1 - .999 ** t)) + 1e-8), -1., 1.)
            return a, (m, v, t), objective(a, p, a0, m0, mask, beta)
        return step

    def plan(self, member, a_start, a0, m0, mask, beta, steps=60, lr=.05):
        jnp = self.jnp
        a = jnp.asarray(a_start, jnp.float32); z = jnp.zeros_like(a); opt = (z, z, 0)
        args = (self.params[member], jnp.asarray(a0, jnp.float32), jnp.asarray(m0, jnp.float32), jnp.asarray(mask, jnp.float32), float(beta))
        obj = None
        for _ in range(steps):
            a, opt, obj = self._plan_step(a, opt, *args, lr)
        return np.asarray(a, np.float64), float(obj)

    def predict(self, a):
        return np.stack([np.asarray(self.f(p, self.jnp.asarray(np.atleast_2d(a), self.jnp.float32))) for p in self.params])


# ----------------------------------------------------------------------------------------------- contact segments
def contact_segments(editor, F):
    """Per side: [(object, start, end, weight(t), post-mask)] as the ILC candidates define them."""
    ref = editor.setup.reference; fps = editor.baseline.fps; t = np.arange(F, dtype=float)
    smooth = lambda x: .5 - .5 * np.cos(np.pi * np.clip(x, 0, 1))
    out = {}
    for side in ref.sides:
        active = np.asarray(ref.contact_active(side)).reshape(F, -1).any(axis=1)
        parts = np.asarray(ref.contact_part_ids(side)).reshape(F, -1) - 1
        idx = np.flatnonzero(active)
        if not len(idx):
            continue
        runs, a0, prev = [], idx[0], idx[0]
        for i in idx[1:]:
            if i - prev > 11:
                runs.append((a0, prev)); a0 = i
            prev = i
        runs.append((a0, prev))
        segs = []
        for a0, b0 in [(a0, b0) for a0, b0 in runs if b0 - a0 + 1 >= 5]:
            sg = parts[a0:b0 + 1][parts[a0:b0 + 1] >= 0]
            o = int(np.bincount(sg).argmax()) if len(sg) else 0
            start, end = min(F - 2, a0 + 6), min(F - 2, b0 + 6)
            w = np.zeros(F); w[(t >= start) & (t <= end)] = 1
            pre = t < start; w[pre] = smooth((t[pre] - (start - .3 * fps)) / (.3 * fps))
            post = t > end; w[post] = 1 - smooth((t[post] - end) / fps)
            segs.append(dict(o=o, start=int(start), end=int(end), w=w, post=post))
        out[side] = segs
    return out


def damped_pinv_step(J, dp, lam=.01):
    return J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(len(J)), dp)


# ----------------------------------------------------------------------------------------------- modes
def ilc_candidates(editor, learner, C):
    grid = [f'{g}:{m}:{r}:{s}' for g in (.25, .5, .75, 1., 1.25, 1.5, 2.) for m in ('xy', 'xyz') for r in (0., .5, 1.) for s in (0., .5, 1.)]
    grid = [grid[i] for i in np.unique(np.linspace(0, len(grid) - 1, C).round().astype(int))]
    old = editor.options.get('ilc_specs')
    editor.options['ilc_specs'] = grid
    try:
        out = editor._ilc_candidates(learner)
    finally:
        editor.options['ilc_specs'] = old
    if not out:   # too few stable worlds: keep the ILC arm pure (no fallback to upper-SAC samples)
        return none_candidates(editor, C, mode='ilc', reason='ilc skipped: fewer than 16 stable worlds')
    editor._mm_a = [None] * len(out)
    return [(dict(mode='ilc', **info), q, t) for info, q, t in out]


def _record_m0(editor, learner):
    keep = current_rollout(editor, learner)
    F = mismatch_features(editor._object_pose)
    editor._mm_m0 = np.median(F[keep], axis=0)
    return keep


def ensemble_gn_candidates(editor, learner, C, D):
    rng = np.random.default_rng(int(editor.training.seed) * 1000 + int(editor.upper_round))
    keep = _record_m0(editor, learner)
    a0 = getattr(editor, '_mm_a0', np.zeros(D)); ed = editable_mask(editor)
    if len(ed) < D:
        ed = np.r_[ed, np.zeros(D - len(ed), bool)]
    hist = getattr(editor, '_mm_hist', [])
    acts, infos = [], []
    if hist:
        n_probe = C // 3
        steps = [(alpha, mu, b) for b in range(4) for mu in (.01, .1, 1., 10.) for alpha in (.25, .5, 1., 1.5, 2.)][:C - n_probe]
        Js = [fit_jacobian(hist, D, rng=None if b == 0 else rng) for b in range(4)]
        for alpha, mu, b in steps:
            a = np.clip(a0 + alpha * gauss_newton_step(Js[b], editor._mm_m0, ed, mu), -1., 1.)
            acts.append(a); infos.append(dict(mode='ensemble_gn', kind='gauss_newton', alpha=alpha, mu=mu, bootstrap=b))
        pred = float(np.linalg.norm(Js[0][:, ed] @ (acts[0] - a0)[ed] + editor._mm_m0))
    else:
        n_probe, pred = C, None
    for a in probes(a0, ed, n_probe, (.1, .2, .3) if not hist else (.15,), rng):
        acts.append(a); infos.append(dict(mode='ensemble_gn', kind='probe'))
    editor._ilc_feedback = dict(stable_worlds=int(keep.sum()), mismatch_norm=float(np.linalg.norm(editor._mm_m0)),
                                history_rounds=len(hist), editable_dims=int(ed.sum()), predicted_norm_first_step=pred)
    return _realize_all(editor, acts, infos)


def surrogate_candidates(editor, learner, C, D):
    rng = np.random.default_rng(int(editor.training.seed) * 1000 + int(editor.upper_round))
    keep = _record_m0(editor, learner)
    a0 = getattr(editor, '_mm_a0', np.zeros(D)); ed = editable_mask(editor)
    if len(ed) < D:
        ed = np.r_[ed, np.zeros(D - len(ed), bool)]
    hist = getattr(editor, '_mm_hist', [])
    acts, infos, fit_loss = [], [], None
    if hist:
        if getattr(editor, '_mm_surrogate', None) is None:
            editor._mm_surrogate = Surrogate(D, len(editor._mm_m0), seed=int(editor.training.seed))
        S = editor._mm_surrogate
        fit_loss = S.fit(hist, steps=400 if len(hist) == 1 else 200, seed=int(editor.upper_round))
        n_probe = C // 3
        plans = [(k, beta, s) for k in range(S.members) for beta in (0., .01, .1, 1.) for s in range(4)][:C - n_probe]
        for k, beta, s in plans:
            start = a0.copy()
            if s:
                start[ed] += .1 * rng.standard_normal(int(ed.sum()))
            a, obj = S.plan(k, np.clip(start, -1, 1), a0, editor._mm_m0, ed.astype(np.float32), beta)
            acts.append(a); infos.append(dict(mode='surrogate', kind='model_gradient', member=k, beta=beta, start=s, predicted_objective=obj))
    else:
        n_probe = C
    for a in probes(a0, ed, n_probe, (.1, .2, .3) if not hist else (.15,), rng):
        acts.append(a); infos.append(dict(mode='surrogate', kind='probe'))
    editor._ilc_feedback = dict(stable_worlds=int(keep.sum()), mismatch_norm=float(np.linalg.norm(editor._mm_m0)),
                                history_rounds=len(hist), editable_dims=int(ed.sum()), surrogate_fit_loss=fit_loss)
    return _realize_all(editor, acts, infos)


def _realize_all(editor, acts, infos):
    try:
        from flash_chord.retargeting.ttsa_editor import ProposalRejected
    except ImportError:   # CPU unit tests without flash_chord
        ProposalRejected = ValueError
    out, kept = [], []
    for a, info in zip(acts, infos):
        try:
            q, t = realize(editor, a)
        except (ProposalRejected, ValueError) as e:
            q = editor.binding.joint_q.copy(); t = editor.binding.joint_target.copy(); info = dict(info, rejected=str(e)); a = None
        out.append((info, q, t)); kept.append(a)
    editor._mm_a = kept
    return out


def ik_feedback_candidates(editor, learner, C):
    """Object-frame mismatch of the hand (palm pose, contact fingertips) -> corrected IK targets -> IK re-solve.
    Palm: wrist joints are set from the corrected palm pose (rigid). Fingers: damped least squares on the fingertip targets in
    the palm frame, Jacobian by finite differences on the exact Newton hand at the current reference (linear in the gain)."""
    keep = current_rollout(editor, learner)
    bq = editor.binding.joint_q.copy(); F = bq.shape[0]; fps = editor.baseline.fps
    OP = np.nan_to_num(np.asarray(editor._object_pose, np.float64))[:, keep]   # (Fm, Wk, O, 14)
    PP = np.nan_to_num(np.asarray(editor._palm_pose, np.float64))[:, keep]     # (Fm, Wk, H, 7)
    TP = np.nan_to_num(np.asarray(editor._tip_pos, np.float64))[:, keep]       # (Fm, Wk, H, 5, 3)
    Fm = OP.shape[0]
    proj = editor.projector; hands = proj.layout.hands; H = len(hands)
    names = list(proj.layout.scalar_joints.names); qids = list(proj.qids)
    ref_p = np.zeros((F, H * 6, 3)); ref_q = np.zeros((F, H * 6, 4))
    for t in range(F):
        p, r = proj.fk(bq[t]); ref_p[t] = p; ref_q[t] = r.as_quat()
    ref = editor.setup.reference
    segs = contact_segments(editor, F)
    tail = np.clip((F - 1 - np.arange(F)) / 3., 0, 1); tail[0] = 0
    knots = np.unique(np.r_[np.arange(1, F - 1, 5), F - 2])
    import newton as _nt
    b = proj.builder
    per_side, fb = {}, {}
    for h, hand in enumerate(hands):
        side = hand.side
        if side not in segs:
            continue
        cols = [qids[names.index(f'{side}_wrist_{c}')] for c in 'xyz']
        ball = [b.joint_q_start[j] for j, kd in enumerate(b.joint_type) if kd == _nt.JointType.BALL and side in str(b.joint_label[j])][0]
        fq = list(hand.finger_q_ids)
        dP = np.zeros((F, 3)); dR = np.zeros((F, 3)); dTip = np.zeros((F, 5, 3)); wsum = np.zeros(F)
        cpos = np.asarray(ref.contact_pos_w(side)).reshape(F, -1, 3)
        for sg in segs[side]:
            o, w = sg['o'], sg['w']
            for k in range(Fm):
                t = k + 1
                if w[t] <= 0:
                    continue
                Rt = Rotation.from_quat(unit_quat(OP[k, 0, o, 10:14])); pt = OP[k, 0, o, 7:10]
                Ra = Rotation.from_quat(unit_quat(OP[k, :, o, 3:7])); pa = OP[k, :, o, 0:3]
                P_ref = Rt.inv().apply(ref_p[t, h * 6] - pt); Q_ref = Rt.inv() * Rotation.from_quat(ref_q[t, h * 6])
                P_sim = Ra.inv().apply(PP[k, :, h, 0:3] - pa); Q_sim = Ra.inv() * Rotation.from_quat(unit_quat(PP[k, :, h, 3:7]))
                dp = np.median(P_sim, axis=0) - P_ref
                dr = np.median((Q_sim * Q_ref.inv()).as_rotvec(), axis=0)
                T_ref = Rt.inv().apply(ref_p[t, h * 6 + 1:h * 6 + 6] - pt)                                  # (5, 3)
                T_sim = np.median(np.stack([Ra.inv().apply(TP[k, :, h, j] - pa) for j in range(5)], axis=1), axis=0)   # (5, 3)
                T_rig = Rotation.from_rotvec(dr).apply(T_ref - P_ref) + P_ref + dp
                near = (np.linalg.norm(ref_p[t, h * 6 + 1:h * 6 + 6, None] - cpos[t][None], axis=-1).min(axis=1) < .03
                        if cpos.shape[1] else np.zeros(5, bool))
                e = (T_sim - T_rig) * near[:, None]
                # back to world, as a correction (minus the mismatch); object-frame quantities rotate with the target object pose
                dP[t] += w[t] * Rt.apply(dp); dR[t] += w[t] * Rt.apply(dr); dTip[t] += w[t] * Rt.apply(e); wsum[t] += w[t]
        dP = gaussian_filter1d(dP, 1.5, axis=0); dR = gaussian_filter1d(dR, 1.5, axis=0); dTip = gaussian_filter1d(dTip, 1.5, axis=0)
        dP *= np.minimum(1., .08 / np.maximum(np.linalg.norm(dP, axis=-1, keepdims=True), 1e-9))
        dR *= np.minimum(1., .5 / np.maximum(np.linalg.norm(dR, axis=-1, keepdims=True), 1e-9))
        dTip *= np.minimum(1., .03 / np.maximum(np.linalg.norm(dTip, axis=-1, keepdims=True), 1e-9))
        # finger IK, unit gain: fingertip displacement -dTip expressed in the palm frame, damped least squares per knot
        dqf = np.zeros((F, len(fq)))
        for t in knots:
            if not np.abs(dTip[t]).max() > 1e-5:
                continue
            Rp = Rotation.from_quat(ref_q[t, h * 6])
            base = Rp.inv().apply(ref_p[t, h * 6 + 1:h * 6 + 6] - ref_p[t, h * 6]).ravel()
            Jf = np.empty((15, len(fq)))
            for j, qid in enumerate(fq):
                qq = bq[t].copy(); qq[qid] += 1e-3
                p1, r1 = proj.fk(qq)
                Jf[:, j] = (Rotation.from_quat(r1.as_quat()[h * 6]).inv().apply(p1[h * 6 + 1:h * 6 + 6] - p1[h * 6]).ravel() - base) / 1e-3
            dqf[t] = damped_pinv_step(Jf, Rp.inv().apply(-dTip[t]).ravel())
        dqf = np.stack([np.interp(np.arange(F), knots, dqf[knots, j]) for j in range(len(fq))], axis=1)
        per_side[side] = (h, cols, ball, fq, dP, dR, dqf)
        fb[side] = dict(palm_mismatch_max_cm=round(float(np.linalg.norm(dP, axis=-1).max()) * 100, 2),
                        palm_rotation_max_deg=round(float(np.degrees(np.linalg.norm(dR, axis=-1).max())), 1),
                        fingertip_mismatch_max_cm=round(float(np.linalg.norm(dTip, axis=-1).max()) * 100, 2))
    lo, hi = proj.lower, proj.upper
    bound = float(editor.options.get('wrist_translation_offset_m', .08)); fbound = float(editor.options.get('finger_offset_rad', .52))
    # no all-zero entry (the unchanged reference can pass the 2-SE gate on replica noise) and no xy/xyz duplicates when gp = 0
    grid = [(gp, gr, gf, m) for gp in (0., .5, 1., 1.5) for gr in (0., .5, 1.) for gf in (0., .5, 1., 1.5, 2.)
            for m in (('xy',) if gp == 0 else ('xy', 'xyz')) if gp or gr or gf]
    grid += [(2., gr, gf, m) for gr in (0., .5, 1.) for gf in (0., 1., 2.) for m in ('xy', 'xyz')]
    grid = grid[:C]
    out = []
    for gp, gr, gf, mode in grid:
        q = bq.copy()
        for side, (h, cols, ball, fq, dP, dR, dqf) in per_side.items():
            d = -gp * dP * tail[:, None]
            if mode == 'xy':
                d[:, 2] = 0
            D = Rotation.from_rotvec(-gr * dR * tail[:, None])
            palm = ref_p[:, h * 6]
            q[:, cols] = palm + d + D.apply(q[:, cols] - palm)
            q[:, ball:ball + 4] = (D * Rotation.from_quat(q[:, ball:ball + 4])).as_quat()
            dev = q[:, cols] - editor.baseline.joint_q[:, cols]
            q[:, cols] = editor.baseline.joint_q[:, cols] + dev * np.minimum(1., bound / np.maximum(np.linalg.norm(dev, axis=1, keepdims=True), 1e-9))
            for j, qid in enumerate(fq):
                i = qids.index(qid)
                v = np.clip(q[:, qid] + gf * dqf[:, j] * tail, editor.baseline.joint_q[:, qid] - fbound, editor.baseline.joint_q[:, qid] + fbound)
                q[:, qid] = np.clip(v, lo[i], hi[i])
        q[0] = bq[0]; q[-1] = bq[-1]
        out.append((dict(mode='ik_feedback', palm_gain=gp, rotation_gain=gr, fingertip_gain=gf, translation=mode,
                         max_change=float(np.abs(q - bq).max())), q, targets_for(editor, q)))
    editor._mm_a = [None] * len(out)
    editor._ilc_feedback = dict(stable_worlds=int(keep.sum()), object_frame_mismatch=fb)
    return out


def none_candidates(editor, C, mode='none', reason=None):
    bq, bt = editor.binding.joint_q.copy(), editor.binding.joint_target.copy()
    editor._mm_a = [None] * C
    editor._ilc_feedback = dict(getattr(editor, '_ilc_feedback', None) or {}, mode=mode, **({'reason': reason} if reason else {}))
    return [(dict(mode=mode, kind='unchanged'), bq.copy(), bt.copy()) for _ in range(C)]


def propose(editor, learner, mode, C, D):
    if mode not in MODES or mode in ('search', 'search_only'):
        raise ValueError(f'propose() does not handle upper_update={mode}')
    editor._mm_mode = mode
    if mode == 'none':
        return none_candidates(editor, C)
    if mode == 'ilc':
        return ilc_candidates(editor, learner, C)
    if mode == 'ensemble_gn':
        return ensemble_gn_candidates(editor, learner, C, D)
    if mode == 'surrogate':
        return surrogate_candidates(editor, learner, C, D)
    if mode == 'surrogate2':
        return surrogate2_candidates(editor, learner, C, D)
    return ik_feedback_candidates(editor, learner, C)


def observe(editor, C, R, NB, costs=None, metrics=None):
    """After the candidate rollout: store (action offset, mismatch offset) pairs for the regression-based modes.
    costs (per world, as the gate scores them) are passed by builds from 2026-10-28 on; surrogate2 needs them."""
    if getattr(editor, '_mm_mode', None) == 'surrogate2':
        return surrogate2_observe(editor, C, R, NB, costs)
    if getattr(editor, '_mm_mode', None) not in ('ensemble_gn', 'surrogate'):
        return
    Fe = mismatch_features(editor._object_pose)
    base = np.median(Fe[C * R:], axis=0)
    a0 = getattr(editor, '_mm_a0', np.zeros(len(next(a for a in editor._mm_a if a is not None))))
    A, M = [], []
    for c, a in enumerate(editor._mm_a[:C]):
        if a is None:
            continue
        A.append(a); M.append(np.median(Fe[c * R:(c + 1) * R], axis=0) - base)
    if A:
        A = np.asarray(A); M = np.asarray(M)
        editor._mm_hist = (getattr(editor, '_mm_hist', []) + [dict(a0=a0.copy(), A=A, dA=A - a0, dM=M, round=int(editor.upper_round))])[-20:]


def commit(editor, selected):
    if getattr(editor, '_mm_mode', None) == 'surrogate2':
        surrogate2_trust_update(editor, selected)
    a = (getattr(editor, '_mm_a', None) or [None])[selected] if selected is not None and selected < len(getattr(editor, '_mm_a', [])) else None
    if a is not None:
        editor._mm_a0 = np.asarray(a, np.float64).copy()


# ----------------------------------------------------------------------------------------------- surrogate v2
# v1 (surrogate) predicts the whole 450-D mismatch trajectory, which after the first failure is mostly chaos (object dropped,
# features clipped), and plans on that; on ep36 its predictions ranked its own candidates at chance (Spearman ~0).
# v2 changes, each logged so the effect can be checked:
#  1. predict what the gate judges: a second head predicts the candidate's cost change vs the baseline replicas;
#     half the model candidates plan on predicted cost, half on mismatch, so the two targets compete inside one run
#  2. mismatch only where an edit can matter: frames up to first failure + 1 s weigh 1, later frames 0.1
#  3. noisy candidates count less: each candidate weighs 1 / (SE^2 of its replica-mean cost + floor)
#  4. trust region: model plans stay within +-rho of the current point; rho x1.5 when a model plan wins, /1.3 when the model
#     misleads (its candidates' mean measured cost change is worse than the random probes'), unchanged otherwise
#  5. self-check every round: Spearman(predicted, measured) cost change of the model candidates, next to the noise ceiling
#     (split-half Spearman of the candidates' own replica costs); written to training/surrogate_quality.jsonl
#  6. every round's (action, cost, mismatch) data saved to training/surrogate_data/rNNN.npz for offline model work
COST_SCALE = 5.


class Surrogate2(Surrogate):
    """One trunk, two heads: windowed mismatch change (first K outputs) and cost change / COST_SCALE (last output)."""

    def __init__(self, D, K, members=5, hidden=256, seed=0):
        self.Km = K
        super().__init__(D, K + 1, members, hidden, seed)

    def _make_train_step(self):
        jax, jnp = self.jax, self.jnp

        def loss(p, A, A0, Y, w, ow):
            d = self.f(p, A) - self.f(p, A0) - Y
            return (jnp.sum(w[:, None] * ow[None] * d ** 2) / jnp.sum(w) / jnp.sum(ow)
                    + 1e-5 * sum(jnp.sum(x ** 2) for x in jax.tree_util.tree_leaves(p)))

        def step(p, opt, A, A0, Y, w, ow, lr):
            g = jax.grad(loss)(p, A, A0, Y, w, ow)
            m, v, t = opt
            t = t + 1
            m = jax.tree_util.tree_map(lambda a, b: .9 * a + .1 * b, m, g)
            v = jax.tree_util.tree_map(lambda a, b: .999 * a + .001 * b * b, v, g)
            p = jax.tree_util.tree_map(lambda x, a, b: x - lr * (a / (1 - .9 ** t)) / (jnp.sqrt(b / (1 - .999 ** t)) + 1e-8), p, m, v)
            return p, (m, v, t), loss(p, A, A0, Y, w, ow)
        return step

    def fit(self, history, frame_w, steps=200, decay=.7, lr=1e-3, seed=0):
        jnp = self.jnp
        rng = np.random.default_rng(seed)
        A = np.concatenate([h['A'] for h in history]); A0 = np.concatenate([np.repeat(h['a0'][None], len(h['A']), 0) for h in history])
        Y = np.concatenate([np.c_[h['dM'], h['dcost']] for h in history])
        w = np.concatenate([decay ** (len(history) - 1 - i) / (h['se'] ** 2 + .05) for i, h in enumerate(history)])
        ow = np.r_[frame_w, frame_w.sum()]            # cost head = half of the loss
        losses = []
        for k in range(self.members):
            i = rng.integers(0, len(A), len(A))
            args = [jnp.asarray(x[i], jnp.float32) for x in (A, A0, Y, w)] + [jnp.asarray(ow, jnp.float32)]
            if self.opt[k] is None:
                z = self.jax.tree_util.tree_map(jnp.zeros_like, self.params[k]); self.opt[k] = (z, z, 0)
            for _ in range(steps):
                self.params[k], self.opt[k], l = self._train_step(self.params[k], self.opt[k], *args, lr)
            losses.append(float(l))
        return losses

    def _make_plan_step(self):
        jax, jnp = self.jax, self.jnp

        def objective(a, p, a0, m0, fw, mask, beta, use_cost):
            d = self.f(p, a) - self.f(p, a0)
            mm = jnp.sum(fw * (m0 + d[:-1]) ** 2) / jnp.sum(fw) - jnp.sum(fw * m0 ** 2) / jnp.sum(fw)   # change vs now
            obj = use_cost * d[-1] + (1. - use_cost) * mm
            return obj + beta * jnp.sum(mask * (a - a0) ** 2) / jnp.maximum(jnp.sum(mask), 1.)

        def step(a, opt, p, a0, m0, fw, mask, beta, use_cost, rho, lr):
            g = jax.grad(objective)(a, p, a0, m0, fw, mask, beta, use_cost) * mask
            m, v, t = opt
            t = t + 1
            m = .9 * m + .1 * g; v = .999 * v + .001 * g * g
            a = a - lr * (m / (1 - .9 ** t)) / (jnp.sqrt(v / (1 - .999 ** t)) + 1e-8)
            a = jnp.clip(jnp.clip(a, a0 - rho, a0 + rho), -1., 1.)
            return a, (m, v, t), objective(a, p, a0, m0, fw, mask, beta, use_cost)
        return step

    def plan(self, member, a_start, a0, m0, fw, mask, beta, use_cost, rho, steps=60, lr=.05):
        jnp = self.jnp
        a = jnp.asarray(a_start, jnp.float32); z = jnp.zeros_like(a); opt = (z, z, 0)
        args = (self.params[member], jnp.asarray(a0, jnp.float32), jnp.asarray(m0, jnp.float32), jnp.asarray(fw, jnp.float32),
                jnp.asarray(mask, jnp.float32), float(beta), float(use_cost), float(rho))
        obj = None
        for _ in range(steps):
            a, opt, obj = self._plan_step(a, opt, *args, lr)
        return np.asarray(a, np.float64), float(obj)

    def predicted_dcost(self, A, a0):
        """Ensemble mean and std of the predicted cost change (in cost units) for actions A vs a0."""
        jnp = self.jnp
        A = jnp.asarray(np.atleast_2d(A), jnp.float32); a0 = jnp.asarray(np.atleast_2d(a0), jnp.float32)
        d = np.stack([np.asarray(self.f(p, A)[:, -1] - self.f(p, a0)[:, -1]) for p in self.params]) * COST_SCALE
        return d.mean(0), d.std(0)


def window_frame_weights(editor, K, stride=2, late=.1):
    """Per-feature weights: frames up to first failure + 1 s count 1, later frames `late` (an edit cannot fix what follows a drop)."""
    O = int(np.asarray(editor._object_pose).shape[2]); per = O * 6
    fr = (np.arange(K) // per) * stride
    tf = (getattr(editor, '_win_info', None) or {}).get('first_fail_frame')
    if tf is None:
        return np.ones(K)
    return np.where(fr <= int(tf) + float(editor.baseline.fps), 1., late)


def surrogate2_candidates(editor, learner, C, D):
    rng = np.random.default_rng(int(editor.training.seed) * 1000 + int(editor.upper_round))
    keep = _record_m0(editor, learner)
    a0 = getattr(editor, '_mm_a0', np.zeros(D)); ed = editable_mask(editor)
    if len(ed) < D:
        ed = np.r_[ed, np.zeros(D - len(ed), bool)]
    hist = getattr(editor, '_mm_hist', [])
    rho = float(getattr(editor, '_sv_rho', .5))
    fw = window_frame_weights(editor, len(editor._mm_m0))
    acts, infos, fit_loss = [], [], None
    if hist:
        if getattr(editor, '_mm_surrogate', None) is None:
            editor._mm_surrogate = Surrogate2(D, len(editor._mm_m0), seed=int(editor.training.seed))
        S = editor._mm_surrogate
        fit_loss = S.fit(hist, fw, steps=400 if len(hist) == 1 else 200, seed=int(editor.upper_round))
        n_probe = C // 3
        plans = [(k, uc, beta, s) for s in range(4) for beta in (0., .1) for k in range(S.members) for uc in (1., 0.)][:C - n_probe]   # interleaved: any cut stays balanced
        for k, uc, beta, s in plans:
            start = a0.copy()
            if s:
                start[ed] += min(.1, rho / 2) * rng.standard_normal(int(ed.sum()))
            a, obj = S.plan(k, np.clip(start, a0 - rho, a0 + rho).clip(-1, 1), a0, editor._mm_m0, fw, ed.astype(np.float32), beta, uc, rho)
            acts.append(a); infos.append(dict(mode='surrogate2', kind='model_cost' if uc else 'model_mismatch', member=k, beta=beta, start=s,
                                              rho=round(rho, 3), predicted_objective=obj))
        mu, sd = S.predicted_dcost(np.asarray(acts), a0)
        for inf, m_, s_ in zip(infos, mu, sd):
            inf.update(predicted_dcost=round(float(m_), 4), predicted_dcost_std=round(float(s_), 4))
    else:
        n_probe = C
    for a in probes(a0, ed, n_probe, (.1, .2, .3) if not hist else (.15,), rng):
        acts.append(a); infos.append(dict(mode='surrogate2', kind='probe'))
    editor._ilc_feedback = dict(stable_worlds=int(keep.sum()), mismatch_norm=float(np.linalg.norm(editor._mm_m0)),
                                history_rounds=len(hist), editable_dims=int(ed.sum()), trust_radius=round(rho, 3),
                                window_frames_weighted=int((fw == 1).sum() // max(1, int(np.asarray(editor._object_pose).shape[2]) * 6)),
                                surrogate_fit_loss=fit_loss)
    editor._sv_infos = infos
    return _realize_all(editor, acts, infos)


def surrogate2_observe(editor, C, R, NB, costs):
    if costs is None:
        raise RuntimeError('surrogate2 needs per-world costs: rebuild the arm with the current patch_mismatch_upper.py')
    from scipy.stats import spearmanr
    costs = np.asarray(costs, np.float64)
    Fe = mismatch_features(editor._object_pose)
    base = np.median(Fe[C * R:], axis=0); bcost = float(costs[C * R:].mean())
    a0 = getattr(editor, '_mm_a0', np.zeros(len(next(a for a in editor._mm_a if a is not None))))
    idx = [c for c, a in enumerate(editor._mm_a[:C]) if a is not None]
    if not idx:
        return
    G = costs[:C * R].reshape(C, R)
    A = np.asarray([editor._mm_a[c] for c in idx]); M = np.asarray([np.median(Fe[c * R:(c + 1) * R], axis=0) - base for c in idx])
    dcost = np.clip((G[idx].mean(1) - bcost) / COST_SCALE, -10., 10.); se = G[idx].std(1, ddof=1) / np.sqrt(R) / COST_SCALE
    editor._mm_hist = (getattr(editor, '_mm_hist', []) + [dict(a0=a0.copy(), A=A, dA=A - a0, dM=M, dcost=dcost, se=se,
                                                               round=int(editor.upper_round))])[-20:]
    # self-check: did the model rank its own candidates right, and how well could ANY model do given replica noise?
    infos = getattr(editor, '_sv_infos', []) or []
    mod = [c for c in idx if c < len(infos) and infos[c].get('kind', '').startswith('model')]
    rec = dict(round=int(editor.upper_round), trust_radius=float(getattr(editor, '_sv_rho', .5)), baseline_cost=round(bcost, 3),
               candidates=len(idx), model_candidates=len(mod))
    h = R // 2
    if len(idx) >= 5:
        rec['noise_ceiling_spearman'] = round(float(spearmanr(G[idx, :h].mean(1), G[idx, h:].mean(1))[0]), 3)
    for kind in ('model_cost', 'model_mismatch'):
        ks = [c for c in mod if infos[c]['kind'] == kind]
        if len(ks) >= 5 and np.std([infos[c]['predicted_dcost'] for c in ks]) > 1e-9:
            rec[f'{kind}_spearman'] = round(float(spearmanr([infos[c]['predicted_dcost'] for c in ks], G[ks].mean(1))[0]), 3)
        if len(ks):
            rec[f'{kind}_mean_dcost'] = round(float(G[ks].mean() - bcost), 3)
    pr = [c for c in idx if c < len(infos) and infos[c].get('kind') == 'probe']
    if pr:
        rec['probe_mean_dcost'] = round(float(G[pr].mean() - bcost), 3)
    if mod and pr:
        rec['model_misleads'] = bool(G[mod].mean() > G[pr].mean())
    editor._sv_last_quality = rec
    print(f'[surrogate2] {rec}', flush=True)
    try:
        out = editor.output.parent
        with (out / 'surrogate_quality.jsonl').open('a') as f:
            f.write(json.dumps(rec) + '\n')
        (out / 'surrogate_data').mkdir(exist_ok=True)
        np.savez_compressed(out / 'surrogate_data' / f'r{int(editor.upper_round):03d}.npz', A=A, a0=a0, dM=M.astype(np.float32),
                            cand_costs=G[idx].astype(np.float32), base_costs=costs[C * R:].astype(np.float32),
                            kinds=np.asarray([infos[c].get('kind', '') if c < len(infos) else '' for c in idx]),
                            predicted_dcost=np.asarray([infos[c].get('predicted_dcost', np.nan) if c < len(infos) else np.nan for c in idx]),
                            frame_w=window_frame_weights(editor, M.shape[1]))
    except Exception as e:
        print(f'[surrogate2] could not save round data: {e}', flush=True)


def surrogate2_trust_update(editor, selected):
    infos = getattr(editor, '_sv_infos', []) or []
    rho = float(getattr(editor, '_sv_rho', .5))
    won = selected is not None and selected < len(infos) and infos[selected].get('kind', '').startswith('model')
    misleads = bool((getattr(editor, '_sv_last_quality', None) or {}).get('model_misleads', False))
    editor._sv_rho = min(1., rho * 1.5) if won else max(.05, rho / 1.3) if misleads else rho
