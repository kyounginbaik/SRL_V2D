"""Archive command provenance stays strict for new and legacy snapshots."""
from dataclasses import fields
import hashlib
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from flash_chord.embodiments.binding import BoundHandReference, RobotReferenceBinding
from flash_chord.retargeting.demo_archive import DemoArchive
from flash_chord.retargeting.reference_snapshot import command_digest, load_snapshot, save_snapshot


def case():
    quat = np.tile([1., 0., 0., 0.], (2, 1))
    hand = BoundHandReference('left', np.zeros((2, 3)), quat, np.zeros((2, 0)),
        np.zeros((2, 1)), np.zeros((2, 1, 3)), quat[:, None],
        np.zeros((2, 1, 3)), quat[:, None])
    binding = RobotReferenceBinding(2, 20., 1, 1, np.zeros((2, 1)), np.zeros((2, 1)), (hand,))
    layout = NS(num_joint_q=1, num_joint_dof=1, sides=('left',),
        scalar_joints=NS(names=('finger',)),
        hand=lambda side: NS(arm_q_ids=(), finger_q_ids=(0,), dp_frames=(0,), fingertip_frames=(0,)))
    values = dict(position=np.zeros((2, 1, 3)), quaternion=quat[:, None],
        contact_position=np.zeros((2, 1, 3)), contact_normal=np.zeros((2, 1, 3)),
        contact_ids=np.zeros((2, 1), dtype=np.int32), active=np.zeros((2, 1), dtype=bool))
    source = NS(sides=('left',), object_body_pos_w=lambda: values['position'],
        object_body_quat_w=lambda: values['quaternion'], contact_pos_w=lambda side: values['contact_position'],
        contact_normal_w=lambda side: values['contact_normal'], contact_part_ids=lambda side: values['contact_ids'],
        contact_active=lambda side: values['active'])
    return binding, layout, source, values


def legacy(tmp_path):
    binding, layout, source, values = case()
    path = tmp_path/'reference.npz'
    provenance = dict(step=100, version=2, origin='baseline', initial_state_sha256='initial-state')
    checksum = save_snapshot(path, binding, layout, provenance=provenance)
    pair = dict(reference='/original/archive/reference.npz', reference_sha256=checksum,
        environment_steps=100, reference_version=2, origin='baseline',
        initial_state_sha256='initial-state', object_command_sha256=command_digest(source))
    (tmp_path/'pair.json').write_text(json.dumps(pair))
    return path, binding, layout, source, values, pair


def test_new_demo_archive_roundtrips_with_source_command_provenance(tmp_path):
    binding, layout, source, _ = case()
    editor = NS(output=tmp_path/'references', config={'task': {}}, metadata={},
        projector=NS(layout=layout), initial_state_sha256='initial-state',
        command_sha256=command_digest(source), version=2, baseline=binding)
    learner = NS(actor=NS(params={'weight': np.ones(2, dtype=np.float32)}, batch_stats={}))
    pair = DemoArchive(editor)._save_pair(binding, learner, np.zeros((2, 1)), np.zeros((2, 1)),
        {}, 1., 100, 'baseline', {'current_success_prefix_frames': 2})
    with np.load(pair['reference'], allow_pickle=False) as data:
        assert json.loads(str(data['header']))['provenance']['object_contact_sha256'] == command_digest(source)
    # New snapshots are self-contained; no legacy sidecar lookup is needed.
    from pathlib import Path
    Path(pair['directory'], 'pair.json').unlink()
    restored = load_snapshot(pair['reference'], layout, binding,
        sha256=pair['reference_sha256'], source_reference=source)
    np.testing.assert_array_equal(restored.joint_q, binding.joint_q)


def test_legacy_archive_recovers_verified_source_and_preserves_every_reference_array(tmp_path):
    path, binding, layout, source, _, pair = legacy(tmp_path)
    contents = path.read_bytes()
    restored = load_snapshot(path, layout, binding, sha256=pair['reference_sha256'], source_reference=source)
    assert path.read_bytes() == contents
    np.testing.assert_array_equal(restored.joint_q, binding.joint_q)
    np.testing.assert_array_equal(restored.joint_target, binding.joint_target)
    for field in fields(BoundHandReference):
        if field.name != 'side':
            np.testing.assert_array_equal(getattr(restored.hands[0], field.name), getattr(binding.hands[0], field.name))


@pytest.mark.parametrize('key,value', [
    ('reference_sha256', '0'*64), ('reference', 'another.npz'),
    ('environment_steps', 101), ('reference_version', 3), ('origin', 'candidate'),
    ('initial_state_sha256', 'other-state'), ('object_command_sha256', '0'*64),
    ('object_command_sha256', None),
])
def test_legacy_archive_rejects_unbound_or_mismatched_sidecar(tmp_path, key, value):
    path, binding, layout, source, _, pair = legacy(tmp_path)
    pair[key] = value
    path.with_name('pair.json').write_text(json.dumps(pair))
    with pytest.raises(ValueError):
        load_snapshot(path, layout, binding, source_reference=source)


@pytest.mark.parametrize('field', ['position', 'quaternion', 'contact_position', 'contact_normal', 'contact_ids', 'active'])
def test_legacy_archive_rejects_actual_changed_source_commands(tmp_path, field):
    path, binding, layout, source, values, _ = legacy(tmp_path)
    values[field] = np.ones_like(values[field])
    with pytest.raises(ValueError, match='source commands do not match'):
        load_snapshot(path, layout, binding, source_reference=source)


def test_missing_provenance_without_archive_record_still_fails(tmp_path):
    path, binding, layout, source, _, _ = legacy(tmp_path)
    path.with_name('pair.json').unlink()
    with pytest.raises(ValueError, match='missing provenance'):
        load_snapshot(path, layout, binding, source_reference=source)


def test_existing_bad_header_digest_cannot_be_overridden_by_sidecar(tmp_path):
    path, binding, layout, source, _, pair = legacy(tmp_path)
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files}
    header = json.loads(str(arrays['header']))
    header['provenance']['object_contact_sha256'] = '0'*64
    arrays['header'] = np.asarray(json.dumps(header))
    np.savez_compressed(path, **arrays)
    pair['reference_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_name('pair.json').write_text(json.dumps(pair))
    with pytest.raises(ValueError, match='source commands do not match'):
        load_snapshot(path, layout, binding, source_reference=source)
