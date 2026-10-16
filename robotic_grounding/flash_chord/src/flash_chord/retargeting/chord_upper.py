"""CHORD-style upper losses on object-centric support, never contact positions.

The squared deviations outside CHORD's relative tolerance band are normalized
by reference support energy. This is an upper search loss, not a verbatim copy
of the paper's exponential RL reward. Objects receive equal final weight.
"""
import numpy as np


class WrenchSupportAccumulator:
    def __init__(self, worlds, hands, objects, basis, *, tolerance=.1, active_threshold=1e-3):
        if not 0 <= tolerance < 1 or active_threshold <= 0:
            raise ValueError('invalid wrench tolerance or active threshold')
        self.shape = (worlds, hands, objects, basis)
        self.tolerance = tolerance
        self.threshold = active_threshold
        shape = (worlds, objects)
        self.band = np.zeros(shape)
        self.missed = np.zeros(shape)
        self.active_count = np.zeros(shape)
        self.unintended = np.zeros(shape)
        self.unintended_energy = np.zeros(shape)
        self.inactive_count = np.zeros(shape)
        self.frames = 0

    def add(self, current, reference):
        current = np.asarray(current, dtype=np.float64).reshape(self.shape)
        reference = np.asarray(reference, dtype=np.float64).reshape(self.shape)
        if (not np.isfinite(current).all() or not np.isfinite(reference).all()
                or np.any(current < 0) or np.any(reference < 0)):
            raise FloatingPointError('invalid object-centric wrench support')
        ref_dirs, cur_dirs = reference > self.threshold, current > self.threshold
        active = ref_dirs.any(axis=-1)
        under = np.maximum((1-self.tolerance)*reference-current, 0.)
        over = np.maximum(current-(1+self.tolerance)*reference, 0.)
        normalized = np.mean(under*under+over*over, axis=-1) / np.maximum(
            np.mean(reference*reference, axis=-1), 1e-6)
        missed = (ref_dirs & ~cur_dirs).sum(axis=-1) / np.maximum(ref_dirs.sum(axis=-1), 1)
        self.band += np.where(active, np.minimum(normalized, 10.), 0.).sum(axis=1)
        self.missed += np.where(active, missed, 0.).sum(axis=1)
        self.active_count += active.sum(axis=1)
        self.unintended += (~active & cur_dirs.any(axis=-1)).sum(axis=1)
        self.unintended_energy += np.where(~active, np.minimum(np.mean(current*current, axis=-1), 10.), 0.).sum(axis=1)
        self.inactive_count += (~active).sum(axis=1)
        self.frames += 1

    def summaries(self):
        if self.frames == 0:
            raise ValueError('no wrench frames collected')
        values = {
            'cws_tolerance_error': self.band / np.maximum(self.active_count, 1),
            'cws_missed_fraction': self.missed / np.maximum(self.active_count, 1),
            'cws_unintended_fraction': self.unintended / np.maximum(self.inactive_count, 1),
            'cws_unintended_energy': self.unintended_energy / np.maximum(self.inactive_count, 1),
        }
        rows = []
        for world in range(self.shape[0]):
            row = {'cws_frames': self.frames}
            for key, value in values.items():
                row[key] = float(value[world].mean())
                row['per_object_'+key] = value[world].tolist()
            rows.append(row)
        return rows


def wrench_contact_gate(candidate, baseline, tolerance=.02):
    """Preserve per-object contact capability without fixing contact locations."""
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError('invalid contact regression tolerance')
    for key in ('per_object_cws_missed_fraction', 'per_object_cws_unintended_fraction'):
        after, before = np.asarray(candidate[key]), np.asarray(baseline[key])
        if (after.shape != before.shape or after.size == 0
                or not np.isfinite(after).all() or not np.isfinite(before).all()
                or np.any(after > before+tolerance)):
            return False
    return True
