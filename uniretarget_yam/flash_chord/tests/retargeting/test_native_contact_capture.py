"""Recorder contract tests using original provider rows and synthetic side fields."""
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import numpy as np
from flash_chord.retargeting.native_contact_capture import NativeContactCapture


class Buffer:
    def __init__(self, data): self.data = np.array(data, copy=True)
    def numpy(self): return self.data


def environment():
    fixture = Path(__file__).parent / 'fixtures/native_contact_wrenches_v56.npz'
    with np.load(fixture) as z:
        p = str(z['frames'][0]) + '__'
        n = len(z[p + 'native_contact__geom'])
        contact = {name: Buffer(z[p + 'native_contact__' + name]) for name in ('geom', 'worldid', 'pos')}
        # These fields test serialization/layout only; no friction/normal truth
        # is claimed for the historical fixture, which did not capture them.
        contact.update(frame=Buffer(np.tile(np.eye(3), (n, 1, 1))),
                       friction=Buffer(np.full((n, 5), .5)), dim=Buffer(np.full(n, 3)), dist=Buffer(np.zeros(n)))
        data = NS(contact=NS(**contact), nacon=Buffer([n]),
                  xpos=Buffer(z[p+'native__xpos']), xquat=Buffer(z[p+'native__xquat']),
                  **{name: Buffer(np.zeros((1, 60))) for name in ('qpos', 'qvel', 'qfrc_constraint', 'qfrc_actuator')})
        contacts = NS(rigid_contact_count=Buffer([n]),
            **{name: Buffer(z[p+'newton_contact__'+name]) for name in ('force', 'rigid_contact_shape0', 'rigid_contact_shape1')})
        return NS(world_count=1, solver=NS(mjw_data=data, mjw_model=NS(geom_bodyid=Buffer(z['geom_bodies'])),
                   mjc_geom_to_newton_shape=Buffer(z['geom_shapes'])), contacts=contacts,
                  model=NS(shape_body=Buffer(z['shape_body'])), command=NS(body_ids_w=Buffer([[52]])))


class CaptureTests(unittest.TestCase):
    def test_original_rows_are_copied_and_ragged_frames_roundtrip(self):
        env = environment(); capture = NativeContactCapture([0])
        original = env.solver.mjw_data.contact.pos.numpy().copy()
        capture(env, 0)
        # Verify capture made no write, and output doesn't alias device buffers.
        np.testing.assert_array_equal(env.solver.mjw_data.contact.pos.numpy(), original)
        env.solver.mjw_data.contact.pos.data[:] += 1
        capture(env, 1)
        result = capture.arrays(); n = len(original)
        np.testing.assert_array_equal(result['native_contact_offsets'], [0, n, 2*n])
        np.testing.assert_array_equal(result['native_contact_pos'][:n], original)
        np.testing.assert_array_equal(result['native_contact_pos'][n:], original+1)
        self.assertEqual(result['native_xpos'].shape[:2], (2, 1))

    def test_force_geometry_row_mismatch_is_rejected(self):
        env=environment(); env.contacts.rigid_contact_shape0.data[0] += 1
        with self.assertRaisesRegex(ValueError, 'correspondence'):
            NativeContactCapture([0])(env, 0)

    def test_stale_contact_conversion_is_rejected(self):
        env=environment(); env.contacts.rigid_contact_count.data[0] -= 1
        with self.assertRaisesRegex(ValueError, 'aligned'):
            NativeContactCapture([0])(env, 0)

    def test_pinned_world_by_geom_map_uses_global_shapes(self):
        env = environment()
        local = env.solver.mjc_geom_to_newton_shape.data
        # Add a second world whose Newton IDs have an arbitrary offset.
        # This catches accidental indexing on the world axis and any modulo
        # assumption about static/replicated shape topology.
        env.world_count = 2
        env.solver.mjc_geom_to_newton_shape = Buffer(np.stack([local, local+200]))
        env.command.body_ids_w = Buffer([[52], [252]])
        env.model.shape_body = Buffer(np.tile(env.model.shape_body.data, 2))
        data = env.solver.mjw_data
        for name in ('xpos', 'xquat', 'qpos', 'qvel', 'qfrc_constraint', 'qfrc_actuator'):
            getattr(data, name).data = np.repeat(getattr(data, name).data, 2, axis=0)
        data.contact.worldid.data[:] = 1
        for side in (0, 1):
            getattr(env.contacts, 'rigid_contact_shape'+str(side)).data += 200
        capture = NativeContactCapture([1]); capture(env, 0)
        self.assertTrue(np.all(capture.arrays()['native_contact_worldid'] == 1))
        # Correspondence in the OTHER world must not authorize these rows.
        env.solver.mjc_geom_to_newton_shape.data[1] -= 200
        with self.assertRaisesRegex(ValueError, 'correspondence'):
            NativeContactCapture([1])(env, 0)

    def test_missing_or_repeated_worlds_are_rejected(self):
        with self.assertRaises(ValueError): NativeContactCapture([0, 0])
        with self.assertRaises(ValueError): NativeContactCapture([1])(environment(), 0)

if __name__ == '__main__': unittest.main()
