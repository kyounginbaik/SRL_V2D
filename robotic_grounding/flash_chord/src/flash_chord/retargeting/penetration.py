"""Hand--object penetration measurements in the Newton collision geometry.

The same contact-pair selector is used for a posed kinematic reference and a
physical rollout. This deliberately measures collision shapes, not rendered
meshes, and never counts robot--support or self contacts as object penetration.
"""

from __future__ import annotations

import numpy as np

from flash_chord.runtime.contact import contact_gaps_w


class HandObjectPenetration:
    def __init__(self, scene, *, tolerance_m: float = 0.002):
        if not 0 <= tolerance_m < 0.02:
            raise ValueError("contact penetration tolerance must be in [0, 0.02) m")
        self.scene = scene
        self.tolerance_m = float(tolerance_m)
        self.shapes_per_world = scene.collision_layout.shape_count
        self.world_count = scene.world_count
        self.object_shapes = set(scene.collision_layout.objects.ids())
        self.sides = tuple(hand.side for hand in scene.collision_layout.hands)
        self.hand_shape_side = {
            shape: side for side, hand in enumerate(scene.collision_layout.hands)
            for shape in hand.contact_shape_ids
        }
        if not self.object_shapes or not self.hand_shape_side:
            raise ValueError("penetration measurement requires object and hand contact shapes")
        self._pipeline = None
        self._contacts = None

    def collide(self, state):
        """Use a dedicated, generously sized buffer; never truncate a collision query."""
        if self._pipeline is None:
            import newton
            self._pipeline = newton.CollisionPipeline(
                self.scene.model, rigid_contact_max=4096 * self.world_count,
            )
            self._contacts = self._pipeline.contacts()
        self._pipeline.collide(state, self._contacts)
        return self._contacts

    def selected_gaps(self, state, contacts) -> list[tuple[int, int, float]]:
        """World, hand-side index, signed gap for every hand--object contact."""
        count = int(contacts.rigid_contact_count.numpy()[0])
        if not count:
            return []
        if count > contacts.rigid_contact_shape0.shape[0]:
            raise RuntimeError(f"penetration contact buffer overflow: {count} > {contacts.rigid_contact_shape0.shape[0]}")
        shape0 = contacts.rigid_contact_shape0.numpy()[:count]
        shape1 = contacts.rigid_contact_shape1.numpy()[:count]
        gaps = contact_gaps_w(self.scene.model, state, contacts)
        finite_shape_count = self.world_count * self.shapes_per_world
        selected = []
        for first, second, gap in zip(shape0, shape1, gaps, strict=True):
            first, second = int(first), int(second)
            if not (0 <= first < finite_shape_count and 0 <= second < finite_shape_count):
                continue  # global ground and invalid shapes are not hand--object pairs
            world0, local0 = divmod(first, self.shapes_per_world)
            world1, local1 = divmod(second, self.shapes_per_world)
            if world0 != world1:
                continue
            side = self.hand_shape_side.get(local0) if local1 in self.object_shapes else None
            if side is None and local0 in self.object_shapes:
                side = self.hand_shape_side.get(local1)
            if side is None:
                continue
            selected.append((world0, side, float(gap)))
        return selected

    def frame_depths(self, state, contacts, active_sides=None) -> np.ndarray:
        """Maximum excess overlap, shape (world, hand side), in metres.

        Reference-labeled contact permits a small solver/mesh tolerance;
        unlabeled contact permits no penetration. Empty contact lists give zero.
        """
        active = np.zeros((self.world_count, len(self.sides)), dtype=bool) if active_sides is None else np.asarray(active_sides, dtype=bool)
        if active.shape == (len(self.sides),):
            active = np.broadcast_to(active, (self.world_count, len(self.sides)))
        if active.shape != (self.world_count, len(self.sides)):
            raise ValueError(f"active_sides must have shape {(len(self.sides),)} or {(self.world_count, len(self.sides))}")
        depths = np.zeros((self.world_count, len(self.sides)), dtype=np.float64)
        for world0, side, gap in self.selected_gaps(state, contacts):
            depth = max(0.0, -gap - (self.tolerance_m if active[world0, side] else 0.0))
            depths[world0, side] = max(depths[world0, side], depth)
        return depths


def penetration_summary(frame_depths: np.ndarray) -> dict[str, float]:
    """Equal frame/world/side weighting; peak is retained as a safety diagnostic."""
    depths = np.asarray(frame_depths, dtype=np.float64)
    if depths.ndim != 3 or not depths.size or not np.all(np.isfinite(depths)) or np.any(depths < 0):
        raise ValueError("expected finite nonnegative time x world x side depths")
    return {
        "penetration_mean_m": float(depths.mean()),
        "penetration_p95_m": float(np.percentile(depths, 95)),
        "penetration_peak_m": float(depths.max()),
        "penetration_fraction_over_2mm": float(np.mean(depths > 0.002)),
    }


def convex_hull_interior_depth(points, normals, offsets) -> np.ndarray:
    """Positive depth inside a convex hull; zero at or outside its surface."""
    values = np.asarray(points) @ np.asarray(normals).T + np.asarray(offsets)
    return np.maximum(-np.max(values, axis=1), 0.0)


def reference_depths(env, monitor: HandObjectPenetration, contact_active: np.ndarray, *, stride: int = 4):
    """Pose the robot/object at sampled reference frames; do no physics steps."""
    if stride < 1:
        raise ValueError("reference penetration stride must be positive")
    frames = np.unique(np.r_[np.arange(0, len(contact_active), stride), len(contact_active) - 1])
    depths = np.empty((len(frames), monitor.world_count, len(monitor.sides)), dtype=np.float64)
    for slot, frame in enumerate(frames):
        env.reset(frame_id=int(frame))
        contacts = monitor.collide(env.state_0)
        depths[slot] = monitor.frame_depths(env.state_0, contacts, contact_active[frame])
    env.reset(frame_id=0)
    return frames, depths


class VisualHullPenetration:
    """Sample visible hand meshes against a convex proxy of a visible object mesh.

    This is a conservative *visual-overlap proxy*, not Newton's collision depth:
    a convex hull fills holes and concavities in the reconstructed object. It is
    appropriate for the box-shaped tissue pilot, not a general shape metric.
    """

    def __init__(self, scene, *, samples_per_shape: int = 48, tolerance_m: float = 0.002):
        import newton
        from scipy.spatial import ConvexHull
        from scipy.spatial.transform import Rotation

        if len(scene.objects) != 1 or len(scene.objects[0].bodies) != 1:
            raise ValueError("visual-hull pilot requires one rigid object body")
        if samples_per_shape < 4:
            raise ValueError("at least four vertices per visible hand shape are required")
        self.scene = scene
        self.sides = tuple(hand.side for hand in scene.collision_layout.hands)
        self.tolerance_m = tolerance_m
        self._rotation = Rotation
        model = scene.model
        flags = np.asarray(model.shape_flags.numpy())
        types = np.asarray(model.shape_type.numpy())
        bodies = np.asarray(model.shape_body.numpy())
        body_labels = tuple(str(label) for label in model.body_label)
        scales = np.asarray(model.shape_scale.numpy(), dtype=np.float64)
        transforms = np.asarray(model.shape_transform.numpy(), dtype=np.float64)
        visible = int(newton.ShapeFlags.VISIBLE)
        mesh_type = int(newton.GeoType.MESH)

        def local_mesh(shape):
            source = model.shape_source[shape]
            if source is None or not hasattr(source, "vertices"):
                return None
            vertices = np.asarray(source.vertices, dtype=np.float64) * scales[shape]
            transform = transforms[shape]
            vertices = Rotation.from_quat(transform[3:7]).apply(vertices) + transform[:3]
            return source, vertices

        object_binding = scene.objects[0]
        self.object_body = object_binding.root.body_id
        vertices = []
        for shape in object_binding.shapes.ids():
            if bodies[shape] != self.object_body or types[shape] != mesh_type or not (flags[shape] & visible):
                continue
            result = local_mesh(shape)
            if result is None:
                continue
            _, points = result
            vertices.append(points)
        if not vertices:
            raise ValueError("object has no visible mesh to build a visual hull")
        planes = ConvexHull(np.concatenate(vertices)).equations
        self.hull_normals = planes[:, :3] / np.linalg.norm(planes[:, :3], axis=1, keepdims=True)
        self.hull_offsets = planes[:, 3] / np.linalg.norm(planes[:, :3], axis=1)

        rng = np.random.default_rng(20261007)
        self.hand_parts = []
        for hand in scene.collision_layout.hands:
            parts = []
            finger_prefixes = tuple(f"{hand.side}_{digit}" for digit in
                                    ("thumb", "index", "middle", "ring", "pinky"))
            for shape in hand.shape_ids:
                if types[shape] != mesh_type or not (flags[shape] & visible):
                    continue
                if not any(prefix in body_labels[int(bodies[shape])] for prefix in finger_prefixes):
                    continue
                result = local_mesh(shape)
                if result is None:
                    continue
                _, points = result
                selected = rng.choice(len(points), min(samples_per_shape, len(points)), replace=False)
                parts.append((int(bodies[shape]), points[selected]))
            if not parts:
                raise ValueError(f"{hand.side} has no visible mesh vertices to sample")
            self.hand_parts.append(parts)
        self.sample_count_by_side = [sum(len(points) for _, points in parts) for parts in self.hand_parts]

    def frame_depths(self, state) -> np.ndarray:
        bodies = np.asarray(state.body_q.numpy(), dtype=np.float64)
        worlds = self.scene.world_count
        bodies_per_world = len(bodies) // worlds
        depths = np.zeros((worlds, len(self.sides)), dtype=np.float64)
        for world in range(worlds):
            object_pose = bodies[world * bodies_per_world + self.object_body]
            inverse = self._rotation.from_quat(object_pose[3:7]).inv()
            for side, parts in enumerate(self.hand_parts):
                samples = []
                for body_id, local_points in parts:
                    pose = bodies[world * bodies_per_world + body_id]
                    world_points = self._rotation.from_quat(pose[3:7]).apply(local_points) + pose[:3]
                    samples.append(inverse.apply(world_points - object_pose[:3]))
                local_points = np.concatenate(samples)
                # For an interior point, the nearest supporting plane gives
                # its depth inside this convex proxy. Outside points are zero.
                inside_depth = convex_hull_interior_depth(local_points, self.hull_normals, self.hull_offsets)
                depths[world, side] = max(0.0, float(np.max(inside_depth)) - self.tolerance_m)
        return depths


def reference_visual_depths(env, monitor: VisualHullPenetration, *, stride: int = 4):
    frames = np.unique(np.r_[np.arange(0, env.scene.robot_reference.num_frames, stride),
                              env.scene.robot_reference.num_frames - 1])
    depths = np.empty((len(frames), env.world_count, len(monitor.sides)), dtype=np.float64)
    for slot, frame in enumerate(frames):
        env.reset(frame_id=int(frame))
        depths[slot] = monitor.frame_depths(env.state_0)
    env.reset(frame_id=0)
    return frames, depths
