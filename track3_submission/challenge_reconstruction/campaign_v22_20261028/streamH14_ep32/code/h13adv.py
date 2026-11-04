"""Stream H13a: global time-advance of a shipped actor's phase reference (no new data, no pose edit).
The actor output is phi(t/dur) @ W + state @ F (normalized-RBF phase basis on uniform centres). Advancing the phase part by delta(t) seconds is
W'[j] = W(tc_j + delta(tc_j)) (linear interpolation of the weight rows along the centre times; clamped at the ends), so phi(t) @ W' ~= phi(t + delta) @ W.
Registered release primitives keyed on absolute times (*_start_s) are moved earlier by delta at that time. Feedback weights, margins, scene untouched.
spec: '0.9' (frames, constant) or piecewise '0.925@0;1.85@2.65' (frames@seconds; 0.2 s smoothstep ramps between pieces).
Prefix 'w' (e.g. 'w0.9'): only the 12 wrist actuators' phase weights are advanced (finger grasp timing and release times unchanged)."""
import os
import numpy as np


def parse(spec):
    if spec is None or str(spec).strip() in ('', '0', '0.0'): return None
    pcs = []
    for p in str(spec).split(';'):
        f, _, t = p.partition('@'); pcs.append((float(t) if t else 0.0, float(f)))
    return sorted(pcs)


def delta_s(pcs, t, fps, ramp=0.2):
    t = np.asarray(t, float); d = np.full(t.shape, pcs[0][1])
    for (t0, f0), (t1, f1) in zip(pcs[:-1], pcs[1:]):
        u = np.clip((t - t1) / ramp + 0.5, 0, 1); u = u ** 3 * (10 - 15 * u + 6 * u ** 2); d = d + (f1 - f0) * u
    return d / fps


def apply(d, W0, spec):
    """returns (W0', extra dict of shifted *_start_s keys) or (W0, {}) if spec is empty."""
    wrist_only = spec is not None and str(spec).startswith('w')
    pcs = parse(str(spec)[1:] if wrist_only else spec)
    if pcs is None: return W0, {}
    fps = float(d['fps']); dur = float(d['duration_s']); tc = np.asarray(d['centers'], float) * dur
    q = np.clip(tc + delta_s(pcs, tc, fps), tc[0], tc[-1])
    W = np.stack([np.interp(q, tc, W0[:, c]) for c in range(W0.shape[1])], 1).astype(W0.dtype)
    extra = {}
    if wrist_only:
        names = [str(x) for x in d['actuator_names']]
        cols = [i for i, n in enumerate(names) if any(n.endswith(f'_{j}_position') for j in ('pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z'))]
        assert len(cols) == 12, cols
        Wm = W0.copy(); Wm[:, cols] = W[:, cols]; return Wm, extra
    for k in d:
        if str(k).endswith('_start_s') and np.asarray(d[k]).size == 1:
            v = float(d[k]); extra[k] = float(v - delta_s(pcs, np.array([v]), fps)[0])
    return W, extra


def spec_from(cfg):
    return cfg.get('advance') or os.environ.get('H13_ADV') or None
