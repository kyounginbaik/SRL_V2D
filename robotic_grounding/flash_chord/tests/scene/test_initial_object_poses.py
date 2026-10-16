"""Named frame-zero poses preserve commands and later-frame reset behavior."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def scene_factory(monkeypatch):
    import newton
    import trimesh
    import warp as wp
    from flash_chord.assets import SHARPAWAVE_URDF_DIR
    from flash_chord.embodiments.binding import build_robot_reference_binding
    from flash_chord.embodiments.sharpa_hands import SharpaHands, SharpaHandsConfig
    from flash_chord.scene import builder as scene_builder
    from flash_chord.scene.collision import ShapeSpan
    from flash_chord.scene.objects import ObjectBinding, ObjectBodyBinding, ObjectRootBinding

    positions = np.array([[[.1, .2, .3], [.4, .5, .6]],
                          [[.2, .3, .4], [.5, .6, .7]]], dtype=np.float32)
    quaternions = np.broadcast_to([1., 0., 0., 0.], (2, 2, 4)).copy()
    articulation = np.zeros((2, 0), dtype=np.float32)
    reference = SimpleNamespace(num_frames=2, fps=20.,
        object_body_pos_w=lambda: positions, object_body_quat_w=lambda: quaternions,
        object_articulation=lambda: articulation)
    robot = SharpaHands(SharpaHandsConfig(
        left_urdf_path=SHARPAWAVE_URDF_DIR/'left_sharpawave_primitive.urdf',
        right_urdf_path=SHARPAWAVE_URDF_DIR/'right_sharpawave_primitive.urdf'))

    def bind(builder, layout, ref):
        q = np.tile(np.asarray(builder.joint_q, np.float32), (ref.num_frames, 1))
        target = np.tile(np.asarray(builder.joint_target_pos, np.float32), (ref.num_frames, 1))
        return build_robot_reference_binding(builder, layout, q, target, ref.fps)

    def spawn(builder, ref, **kwargs):
        objects = []
        mesh = trimesh.creation.box(extents=(.04, .06, .08))
        for index, name in enumerate(('vessel', 'lid')):
            q, dof, shape = builder.joint_coord_count, builder.joint_dof_count, builder.shape_count
            body = builder.add_link(label=name)
            joint = builder.add_joint_free(child=body, label=name+'_free')
            builder.add_articulation([joint], label=name)
            builder.add_shape_mesh(body, mesh=newton.Mesh(
                np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.faces, dtype=np.int32).reshape(-1)))
            builder.joint_q[q:q+3] = ref.object_body_pos_w()[0, index].tolist()
            root = ObjectRootBinding(body, index, tuple(range(q, q+7)), tuple(range(dof, dof+6)))
            objects.append(ObjectBinding(name, (ObjectBodyBinding(body, index),), root, (),
                                         ShapeSpan(shape, builder.shape_count)))
        return objects

    monkeypatch.setattr(scene_builder, 'spawn_objects', spawn)
    embodiment = SimpleNamespace(build=robot.build, bind_reference=bind)

    def build(**kwargs):
        with wp.ScopedDevice('cpu'):
            return scene_builder.build_scene(embodiment, reference, **kwargs), reference
    return build


def _reset_pair(scene, reference, frame):
    import warp as wp
    from flash_chord.lifecycle.reset import ReferenceResetTable, reset_scene_to_frame, reset_worlds
    with wp.ScopedDevice('cpu'):
        host, runtime = scene.model.state(), scene.model.state()
        table = ReferenceResetTable.build(scene, reference, device='cpu')
        runtime.joint_qd.assign(np.ones(scene.model.joint_dof_count, dtype=np.float32))
        reset_scene_to_frame(scene, reference, frame, host)
        reset_worlds(scene.model, table, runtime, frame_id=frame)
    # Host uses float64 assembly; the device adds scale offsets in float32.
    np.testing.assert_allclose(host.joint_q.numpy(), runtime.joint_q.numpy(), rtol=0, atol=6e-8)
    np.testing.assert_allclose(host.body_q.numpy(), runtime.body_q.numpy(), atol=1e-7)
    np.testing.assert_array_equal(runtime.joint_qd.numpy(), 0.)
    return runtime, table


def _poses():
    # Reverse mapping order, with a nontrivial rotation; logical names bind poses.
    return {'lid': {'position': [.41, .52, .63], 'quaternion_wxyz': [np.cos(.1), np.sin(.1), 0., 0.]},
            'vessel': {'position': [.11, .22, .33], 'quaternion_wxyz': [1., 0., 0., 0.]}}


def test_named_frame_zero_poses_preserve_reference_and_later_resets(scene_factory):
    from flash_chord.retargeting.fixed_start import assert_initial_state
    import warp as wp
    baseline, reference = scene_factory(world_count=2)
    original = [a.copy() for a in (reference.object_body_pos_w(), reference.object_body_quat_w(),
                                   reference.object_articulation())]
    poses = _poses()
    shifted, same_reference = scene_factory(world_count=2, object_initial_pose_overrides=poses)
    np.testing.assert_array_equal(shifted.object_root_position_offsets_w, 0.)
    for frame in (0, 1):
        state, table = _reset_pair(shifted, same_reference, frame)
        q = state.joint_q.numpy().reshape(2, -1)
        body_q = state.body_q.numpy().reshape(2, -1, 7)
        for index, obj in enumerate(shifted.objects):
            if frame == 0:
                p = poses[obj.name]['position']
                wxyz = np.asarray(poses[obj.name]['quaternion_wxyz'])
            else:
                p = reference.object_body_pos_w()[frame, index]
                wxyz = reference.object_body_quat_w()[frame, index]
            expected = np.r_[p, wxyz[[1,2,3,0]]]
            np.testing.assert_allclose(q[:, list(obj.root.free_q_ids)],
                                       np.broadcast_to(expected, (2, 7)), atol=3e-8)
            # Replicated worlds must receive the same reference-world pose once.
            np.testing.assert_allclose(body_q[:, obj.root.body_id],
                                       np.broadcast_to(expected, (2, 7)), atol=3e-8)
            if frame == 0:
                model_q = shifted.model.joint_q.numpy().reshape(2, -1)
                np.testing.assert_allclose(model_q[:, list(obj.root.free_q_ids)],
                                           np.broadcast_to(expected, (2, 7)), atol=3e-8)
    state, table = _reset_pair(shifted, same_reference, 0)
    base_state, base_table = _reset_pair(baseline, reference, 0)
    with wp.ScopedDevice('cpu'):
        env = SimpleNamespace(world_count=2, state_0=state, reset_table=table,
                              timestep=wp.zeros(2, dtype=wp.int32))
        base_env = SimpleNamespace(world_count=2, state_0=base_state, reset_table=base_table,
                                   timestep=wp.zeros(2, dtype=wp.int32))
    actual = assert_initial_state(env)['initial_state_sha256']
    prior = assert_initial_state(base_env)['initial_state_sha256']
    assert actual != prior
    assert assert_initial_state(env, actual)['initial_state_sha256'] == actual
    with pytest.raises(ValueError, match='initial state changed'):
        assert_initial_state(env, prior)
    for before, after in zip(original, (reference.object_body_pos_w(), reference.object_body_quat_w(),
                                        reference.object_articulation()), strict=True):
        np.testing.assert_array_equal(before, after)
    np.testing.assert_array_equal(table.ref_joint_q.numpy()[1:], base_table.ref_joint_q.numpy()[1:])
    np.testing.assert_array_equal(table.ref_joint_q.numpy()[:, :shifted.layout.num_joint_q],
                                  base_table.ref_joint_q.numpy()[:, :shifted.layout.num_joint_q])


def test_initial_poses_preserve_additive_scale_compensation(scene_factory):
    options = dict(world_count=3, object_scale_min=.8, object_scale_max=1.2, object_scale_seed=7)
    baseline, reference = scene_factory(**options)
    poses = _poses()
    translated, _ = scene_factory(**options, object_initial_pose_overrides=poses)
    assert np.any(baseline.object_root_position_offsets_w[1:] != 0.)
    np.testing.assert_array_equal(translated.object_root_position_offsets_w, baseline.object_root_position_offsets_w)
    np.testing.assert_array_equal(translated.object_scales, baseline.object_scales)
    for frame in (0, 1):
        state, table = _reset_pair(translated, reference, frame)
        for world in range(3):
            for index, obj in enumerate(translated.objects):
                root = world*table.per_world_q + obj.root.free_q_ids[0]
                p = np.asarray(poses[obj.name]['position']) if frame == 0 else reference.object_body_pos_w()[frame,index]
                np.testing.assert_allclose(state.joint_q.numpy()[root:root+3],
                    p+translated.object_root_position_offsets_w[world,index], atol=3e-8)


@pytest.mark.parametrize('case', ['empty', 'missing', 'unknown', 'sequence', 'short_position',
    'nan_position', 'infinite_position', 'overflow_position', 'short_quaternion', 'zero_quaternion',
    'nonunit_quaternion', 'extra_field'])
def test_invalid_names_or_poses_are_rejected(case):
    from flash_chord.scene.builder import _named_initial_poses
    poses = _poses()
    if case == 'empty': poses = {}
    elif case == 'missing': del poses['vessel']
    elif case == 'unknown': poses['unknown'] = poses.pop('vessel')
    elif case == 'sequence': poses = list(poses.values())
    elif case == 'short_position': poses['lid']['position'] = [0,0]
    elif case == 'nan_position': poses['lid']['position'] = [0,np.nan,0]
    elif case == 'infinite_position': poses['lid']['position'] = [0,np.inf,0]
    elif case == 'overflow_position': poses['lid']['position'] = [1e100,0,0]
    elif case == 'short_quaternion': poses['lid']['quaternion_wxyz'] = [1,0,0]
    elif case == 'zero_quaternion': poses['lid']['quaternion_wxyz'] = [0,0,0,0]
    elif case == 'nonunit_quaternion': poses['lid']['quaternion_wxyz'] = [2,0,0,0]
    else: poses['lid']['frame'] = 1
    with pytest.raises(ValueError, match='object_initial_pose_overrides'):
        _named_initial_poses([SimpleNamespace(name='vessel'), SimpleNamespace(name='lid')], poses)


def test_config_defaults_none_and_named_mapping_roundtrips():
    from omegaconf import OmegaConf
    from flash_chord.scene.builder import _named_initial_poses
    root = Path(__file__).resolve().parents[2]/'src/flash_chord/configs/scene'
    for name in ('train', 'eval', 'replay'):
        config = OmegaConf.load(root/f'{name}.yaml')
        assert config.object_initial_pose_overrides is None
    poses = _poses()
    for value in poses.values():
        value['quaternion_wxyz'] = [float(x) for x in value['quaternion_wxyz']]
    config.object_initial_pose_overrides = poses
    restored = OmegaConf.create(OmegaConf.to_yaml(config))
    actual = _named_initial_poses([SimpleNamespace(name='vessel'), SimpleNamespace(name='lid')],
                                  restored.object_initial_pose_overrides)
    np.testing.assert_allclose(actual['lid']['quaternion_wxyz'], poses['lid']['quaternion_wxyz'])
