"""Grasp-prototype exploration for the UniRetarget upper level.

The upper level edits finger references per hand and per contact segment (from the immutable
reconstruction contact labels) by blending toward a library of physically validated grasp postures
(DexAgent clusters + GRASP-taxonomy templates), instead of per-joint or PCA offsets.

Per (hand, segment) action, all in [-1, 1]:
  K logits   -> w = softmax(temperature * logits)         which grasp
  alpha_raw  -> alpha = max(0, alpha_raw)                 how strongly (0 = keep reference fingers)
  squeeze_raw-> squeeze = max_squeeze * max(0, squeeze_raw) extra flexion past the posture
Finger target in the window: q = q_ref + alpha * env(t) * (posture - q_ref), env ramps up over a
pre-shape before contact onset and down after release. A zero action leaves fingers unchanged.
"""
import numpy as np


class GraspPrototypes:
    def __init__(self, library_path, reference, joint_names, qids, fps, num_frames, *, preshape_s=.4,
                 release_s=.3, merge_gap=5, min_frames=5, temperature=8.0, max_squeeze=.15,
                 structured_fraction=.25, seed=0):
        lib = np.load(library_path)
        self.names = [str(n) for n in lib['names']]
        self.postures = np.asarray(lib['postures'], np.float64)          # K x 22
        order = [str(j) for j in lib['joint_order']]
        self.K = len(self.names)
        self.temperature, self.max_squeeze = float(temperature), float(max_squeeze)
        self.structured_fraction = float(structured_fraction)
        self.rng = np.random.default_rng(seed)
        self.hands = []
        flexion = np.array([not (j.endswith('AA') or j in ('thumb_CMC_FE', 'pinky_CMC')) for j in order])
        F = int(num_frames)
        t = np.arange(F)
        for side in reference.sides:
            cols = np.array([qids[joint_names.index(f'{side}_{j}')] for j in order])
            active = np.asarray(reference.contact_active(side)).reshape(F, -1).any(axis=1)
            segments = self._segments(active, merge_gap, min_frames)
            envelopes = []
            pre, rel = max(1, int(round(preshape_s * fps))), max(1, int(round(release_s * fps)))
            for start, end in segments:
                env = np.zeros(F)
                env[start:end + 1] = 1.0
                up = (t >= start - pre) & (t < start)
                env[up] = .5 - .5 * np.cos(np.pi * (t[up] - (start - pre)) / pre)
                down = (t > end) & (t <= end + rel)
                env[down] = .5 + .5 * np.cos(np.pi * (t[down] - end) / rel)
                env[[0, -1]] = 0.0  # endpoints stay exactly on the published reference
                envelopes.append(env)
            self.hands.append(dict(side=side, cols=cols, segments=segments, envelopes=envelopes))
        self.flexion = flexion
        self.lower = np.asarray(lib['lower'], np.float64)
        self.upper = np.asarray(lib['upper'], np.float64)
        self.per_segment = self.K + 2
        self.action_dim = sum(len(h['segments']) for h in self.hands) * self.per_segment
        if self.action_dim == 0:
            raise ValueError('grasp prototypes need at least one contact segment')

    @staticmethod
    def _segments(active, merge_gap, min_frames):
        idx = np.flatnonzero(active)
        if not len(idx):
            return []
        runs, start, prev = [], idx[0], idx[0]
        for i in idx[1:]:
            if i - prev > merge_gap + 1:
                runs.append((start, prev))
                start = i
            prev = i
        runs.append((start, prev))
        return [(int(a), int(b)) for a, b in runs if b - a + 1 >= min_frames]

    def decode(self, extra):
        """Per (hand, segment): (weights, alpha, squeeze, posture)."""
        a = np.clip(np.asarray(extra, np.float64), -1, 1).reshape(-1, self.per_segment)
        out, k = [], 0
        for hand in self.hands:
            row = []
            for _ in hand['segments']:
                logits = self.temperature * a[k, :self.K]
                w = np.exp(logits - logits.max())
                w /= w.sum()
                alpha = max(0.0, a[k, self.K])
                squeeze = self.max_squeeze * max(0.0, a[k, self.K + 1])
                posture = w @ self.postures + squeeze * self.flexion
                row.append((w, alpha, squeeze, np.clip(posture, self.lower, self.upper)))
                k += 1
            out.append(row)
        return out

    def finger_delta(self, extra, base_q):
        """Finger-joint correction (frames x num_joint_q), zero outside contact windows."""
        delta = np.zeros_like(base_q, dtype=np.float64)
        for hand, row in zip(self.hands, self.decode(extra)):
            cols = hand['cols']
            for env, (_, alpha, _, posture) in zip(hand['envelopes'], row):
                if alpha <= 0:
                    continue
                proposed = alpha * env[:, None] * (posture[None] - base_q[:, cols])
                current = delta[:, cols]
                # Overlapping windows: keep the stronger edit per frame.
                delta[:, cols] = np.where(np.abs(proposed) > np.abs(current), proposed, current)
        return delta

    def structured(self, actions, offset):
        """Overwrite a fraction of candidates' prototype part with pure single-prototype proposals."""
        actions = np.array(actions, copy=True)
        rows = int(round(self.structured_fraction * len(actions)))
        extra = actions[:rows, offset:].reshape(rows, -1, self.per_segment)
        for r in range(rows):
            for s in range(extra.shape[1]):
                k = self.rng.integers(self.K)
                extra[r, s, :self.K] = -1.0
                extra[r, s, k] = 1.0
                extra[r, s, self.K] = self.rng.choice([.4, .7, 1.0])
                extra[r, s, self.K + 1] = self.rng.uniform(0, 1)
        actions[:rows, offset:] = extra.reshape(rows, -1)
        return actions

    def describe(self, extra):
        rows = []
        for hand, row in zip(self.hands, self.decode(extra)):
            for (start, end), (w, alpha, squeeze, _) in zip(hand['segments'], row):
                rows.append(dict(side=hand['side'], start=start, end=end, prototype=self.names[int(np.argmax(w))],
                                 weight=round(float(w.max()), 3), alpha=round(alpha, 3), squeeze_rad=round(squeeze, 3)))
        return rows
