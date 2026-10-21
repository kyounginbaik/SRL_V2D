"""Distance from simulated fingertips to immutable, object-attached contact labels."""

import numpy as np
from scipy.spatial.transform import Rotation


class ContactProximity:
    """Evaluate labeled hand/object reach without moving the object commands."""

    def __init__(self, reference, sides):
        self.sides = tuple(sides)
        self.num_objects = len(reference.object_body_names())
        self.local = {}
        self.parts = {}
        ref_pos = np.asarray(reference.object_body_pos_w(), dtype=np.float64)
        ref_quat = np.asarray(reference.object_body_quat_w(), dtype=np.float64)[..., [1, 2, 3, 0]]
        if ref_pos.shape[1] != self.num_objects:
            raise ValueError("contact/object body count mismatch")
        for side in self.sides:
            parts = np.asarray(reference.contact_part_ids(side), dtype=np.int64) - 1
            points = np.asarray(reference.contact_pos_w(side), dtype=np.float64)
            if points.shape[:2] != parts.shape or points.shape[0] != ref_pos.shape[0]:
                raise ValueError(f"invalid contact shape for {side}")
            active = parts >= 0
            if np.any(parts[active] >= self.num_objects):
                raise ValueError(f"contact part ID outside object roster for {side}")
            local = np.zeros_like(points)
            if active.any():
                frames, slots = np.nonzero(active)
                bodies = parts[frames, slots]
                local[frames, slots] = Rotation.from_quat(ref_quat[frames, bodies]).inv().apply(
                    points[frames, slots] - ref_pos[frames, bodies]
                )
            self.local[side] = local
            self.parts[side] = parts

    def frame(self, frame_ids, fingertip_positions, actual_object_poses, *, return_vectors=False):
        """Return distance sums/counts and optionally contact-directed vectors (vectorized over worlds)."""
        ids = np.asarray(frame_ids, dtype=np.int64).reshape(-1)
        tips = np.asarray(fingertip_positions, dtype=np.float64)
        objects = np.asarray(actual_object_poses, dtype=np.float64)
        worlds = len(ids)
        if tips.shape[:2] != (worlds, len(self.sides)) or tips.shape[-1] != 3:
            raise ValueError("expected world x hand x fingertip x xyz")
        if objects.shape != (worlds, self.num_objects, 7):
            raise ValueError("expected world x object x Newton pose")
        sums = np.zeros((worlds, len(self.sides), self.num_objects), dtype=np.float64)
        counts = np.zeros_like(sums, dtype=np.int64)
        vectors = np.zeros((*sums.shape, 3), dtype=np.float64) if return_vectors else None
        world_index = np.arange(worlds)[:, None]
        for hand, side in enumerate(self.sides):
            parts = self.parts[side][ids]                      # W x K
            active = parts >= 0
            if not active.any():
                continue
            bodies = np.where(active, parts, 0)
            poses = objects[world_index, bodies]               # W x K x 7
            norm = np.linalg.norm(poses[..., 3:], axis=-1)
            bad_slot = active & (~np.isfinite(poses).all(axis=-1) | (norm < 1e-8))
            bad = bad_slot.any(axis=1) | ~np.isfinite(tips[:, hand]).all(axis=(1, 2))   # per world
            w_bad, k_bad = np.nonzero(active & bad[:, None])
            size = worlds * self.num_objects
            flat = w_bad * self.num_objects + bodies[w_bad, k_bad]
            sums[:, hand] += np.bincount(flat, minlength=size).reshape(worlds, -1) * 2.0
            counts[:, hand] += np.bincount(flat, minlength=size).reshape(worlds, -1)
            good = active & ~bad[:, None]
            w_ok, k_ok = np.nonzero(good)
            if not len(w_ok):
                continue
            pose = poses[w_ok, k_ok]
            quat = pose[:, 3:] / np.linalg.norm(pose[:, 3:], axis=-1, keepdims=True)  # xyzw
            local = self.local[side][ids[w_ok], k_ok]
            u, s = quat[:, :3], quat[:, 3:4]
            uv = np.cross(u, local)
            targets = local + 2.0 * (s * uv + np.cross(u, uv)) + pose[:, :3]
            differences = targets[:, None, :] - tips[w_ok, hand]                    # N x tips x 3
            nearest = np.linalg.norm(differences, axis=-1).argmin(axis=1)
            chosen = differences[np.arange(len(w_ok)), nearest]
            distances = np.linalg.norm(chosen, axis=-1)
            body = bodies[w_ok, k_ok]
            flat = w_ok * self.num_objects + body
            sums[:, hand] += np.bincount(flat, weights=distances, minlength=size).reshape(worlds, -1)
            counts[:, hand] += np.bincount(flat, minlength=size).reshape(worlds, -1)
            if return_vectors:
                for axis in range(3):
                    vectors[:, hand, :, axis] += np.bincount(flat, weights=chosen[:, axis], minlength=size).reshape(worlds, -1)
        return (sums, counts, vectors) if return_vectors else (sums, counts)

def summarize_contact_proximity(sums, counts, *, cap_m=2.0):
    """Equal-weight active objects; preserve hand/object scores for safety gates."""
    total = np.asarray(sums, dtype=np.float64)
    n = np.asarray(counts, dtype=np.int64)
    if total.shape != n.shape or total.ndim != 3:
        raise ValueError("expected matching world x hand x object sums and counts")
    distance = np.divide(total, np.maximum(n, 1))
    per_hand_object = []
    for hand in range(n.shape[1]):
        hand_scores = []
        for body in range(n.shape[2]):
            active = n[:, hand, body] > 0
            hand_scores.append(float(np.clip(distance[active, hand, body], 0, cap_m).mean())
                               if active.any() else None)
        per_hand_object.append(hand_scores)
    object_sums = total.sum(axis=1)
    object_counts = n.sum(axis=1)
    object_distance = np.divide(object_sums, np.maximum(object_counts, 1))
    per_object = []
    for body in range(n.shape[2]):
        active = object_counts[:, body] > 0
        per_object.append(float(np.clip(object_distance[active, body], 0, cap_m).mean())
                          if active.any() else None)
    # Equal hand/object weight keeps a reachable hand informative when the
    # other hand is far away; object-pooled averages can hide that signal.
    present = [value for row in per_hand_object for value in row if value is not None]
    if not present:
        raise ValueError("no labeled contacts were evaluated")
    return dict(contact_proximity_mean_m=float(np.mean(present)),
                per_object_contact_proximity_mean_m=per_object,
                per_hand_object_contact_proximity_mean_m=per_hand_object,
                contact_proximity_active_labels=int(n.sum()))
