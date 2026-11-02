"""Read-only MuJoCo-Warp force-evaluation snapshots for contact optimization.

Contact geometry and cached body poses come from mjw_data, not transported
Newton post-step geometry. This version records evidence only; it does not
assume a normal sign or change any simulator/control buffer.
"""
import numpy as np


def _numpy(value):
    return np.asarray(value.numpy()).copy()


def _prefix_numpy(value,count):
    # Native contact capacity can be worlds*nconmax, orders of magnitude
    # larger than the live contact count. Slice the device view BEFORE the
    # host transfer instead of copying the entire capacity every frame.
    return _numpy(value[:count])


class NativeContactCapture:
    def __init__(self, world_ids):
        self.world_ids = np.asarray(world_ids, np.int32)
        if (self.world_ids.ndim != 1 or len(self.world_ids) == 0
                or np.any(self.world_ids < 0) or len(np.unique(self.world_ids)) != len(self.world_ids)):
            raise ValueError('unique nonnegative selected world IDs required')
        self.records = []
        self.environment = None

    def __call__(self, env, frame):
        if frame != len(self.records):
            raise ValueError('native capture requires contiguous executed control frames')
        solver = env.solver
        data, model = solver.mjw_data, solver.mjw_model
        if self.environment is None:
            if frame != 0 or self.world_ids.max() >= env.world_count:
                raise ValueError('capture must start at frame zero with existing worlds')
            self.environment = env
            self.static = dict(native_capture_world_ids=self.world_ids.copy(),
                native_geom_to_newton_shape=_numpy(solver.mjc_geom_to_newton_shape),
                native_geom_bodyid=_numpy(model.geom_bodyid),
                native_body_to_newton_body=_numpy(solver.mjc_body_to_newton) if hasattr(solver, 'mjc_body_to_newton') else np.empty(0, np.int32),
                native_shape_body=_numpy(env.model.shape_body),
                native_object_body_ids=_numpy(env.command.body_ids_w).reshape(env.world_count, -1)[self.world_ids],
                native_capture_semantics=np.asarray('last control substep; cached MuJoCo force-evaluation geometry; raw native frames'))
        elif self.environment is not env:
            raise ValueError('each capture belongs to exactly one fresh environment')
        count = int(_numpy(data.nacon).reshape(-1)[0])
        converted_count = int(_numpy(env.contacts.rigid_contact_count).reshape(-1)[0])
        if count < 0 or count != converted_count:
            raise ValueError('native and converted contact buffers are not aligned')
        contact = data.contact
        worlds = _prefix_numpy(contact.worldid,count)
        selected = np.isin(worlds, self.world_ids)
        result = {'worldid': worlds[selected]}
        for name in ('geom', 'pos', 'frame', 'friction', 'dim', 'dist'):
            result[name] = _prefix_numpy(getattr(contact, name),count)[selected]
        for name in ('force', 'rigid_contact_shape0', 'rigid_contact_shape1'):
            result['newton_' + name] = _prefix_numpy(getattr(env.contacts, name),count)[selected]
        # Both representations must refer to the same geom rows. Reject an
        # adapter/provider change instead of silently mismatching forces.
        geom = result['geom']
        mapping = self.static['native_geom_to_newton_shape']
        if mapping.ndim == 2 and mapping.shape[0] == env.world_count:
            ngeom = mapping.shape[1]
        elif mapping.ndim == 1:
            # Historical fixtures use local shape IDs for a shared model.
            ngeom = len(mapping)
        else:
            raise ValueError('unsupported native geom-to-shape map layout')
        if (geom.shape != (selected.sum(), 2) or np.any(geom < 0)
                or np.any(geom >= ngeom)):
            raise ValueError('unsupported native contact geom layout')
        for side in (0, 1):
            shapes = result['newton_rigid_contact_shape' + str(side)]
            if mapping.ndim == 2:
                # Pinned provider maps [world, local geom] to GLOBAL Newton
                # shape IDs, including shared static shapes. Never modulo.
                native_shapes = mapping[worlds[selected], geom[:, side]]
                matches = native_shapes == shapes
            else:
                shape_count = len(self.static['native_shape_body']) // env.world_count
                if shape_count <= 0:
                    raise ValueError('invalid replicated shape count')
                matches = mapping[geom[:, side]] == shapes % shape_count
            if not np.all(matches):
                raise ValueError('native geom and Newton shape correspondence changed')
        for name in ('xpos', 'xquat', 'qfrc_constraint', 'qfrc_actuator', 'qpos', 'qvel', 'cdof', 'subtree_com'):
            if name in ('cdof', 'subtree_com') and not hasattr(data, name):
                continue  # Historical fixture provider omitted these caches.
            values = _numpy(getattr(data, name))
            if values.shape[0] != env.world_count:
                raise ValueError('unsupported native world-major ' + name)
            result[name] = values[self.world_ids]
        if not all(np.isfinite(values).all() for values in result.values()):
            raise ValueError('nonfinite native contact snapshot')
        self.records.append(result)

    def arrays(self):
        if not self.records:
            raise ValueError('no native force snapshots captured')
        result = dict(self.static)
        lengths = [len(row['worldid']) for row in self.records]
        result['native_contact_offsets'] = np.cumsum([0] + lengths)
        result['native_capture_frame_ids'] = np.arange(len(self.records))
        for key in self.records[0]:
            if key in ('xpos', 'xquat', 'qfrc_constraint', 'qfrc_actuator', 'qpos', 'qvel', 'cdof', 'subtree_com'):
                result['native_' + key] = np.stack([row[key] for row in self.records])
            else:
                result['native_contact_' + key] = np.concatenate([row[key] for row in self.records])
        return result
