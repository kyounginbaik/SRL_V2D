"""Measured support orientation survives standalone asset resolution and spawning."""
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize('oriented', [False, True])
def test_colocated_measured_table_orientation(tmp_path, oriented):
    import newton
    import warp as wp
    from flash_chord.assets.registry import support_usda_for_reference
    from flash_chord.scene.support import add_support_surfaces
    from flash_chord.data.reference import ReferenceMetadata

    assets = tmp_path/'assets'
    assets.mkdir()
    support = assets/'support.usda'
    angle = np.deg2rad(9.)
    w, x = np.cos(angle/2), np.sin(angle/2)
    rotation = f'quatd xformOp:orient = ({w}, {x}, 0, 0)' if oriented else ''
    support.write_text('''#usda 1.0
def Cylinder "table" {
    double radius = 0.3
    double height = 0.04
    double3 xformOp:translate = (0.1, 0.2, 0.3)
    '''+rotation+'''
}
''')
    reference = SimpleNamespace(metadata=ReferenceMetadata(str(tmp_path/'reference.parquet'), 20.))
    assert support_usda_for_reference(reference) == support
    with wp.ScopedDevice('cpu'):
        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        binding = add_support_surfaces(builder, support)
    assert binding.body_id == 0
    transform = np.asarray(builder.shape_transform[0])
    np.testing.assert_allclose(transform[:3], [.1,.2,.3], atol=1e-7)
    expected = [x,0,0,w] if oriented else [0,0,0,1]
    np.testing.assert_allclose(transform[3:], expected, atol=1e-7)
    # This is a collision support, not a dynamic object or a pose offset.
    assert builder.body_label == ['support']
    assert len(builder.shape_body) == 1
    assert binding.free_q_ids == binding.free_dof_ids == ()
    assert builder.joint_coord_count == builder.joint_dof_count == 0
    assert builder.joint_type == [newton.JointType.FIXED]
    assert builder.joint_parent == [-1]
    assert builder.joint_child == [binding.body_id]


@pytest.mark.parametrize('coefficients', [
    (0., 0., 1.),
    (0., 0., 1., 0., 0.),
    (0., 0., 0., 1.),
    (np.nan, 0., 1., 0.),
    (0., 0., np.inf, 0.),
    (0., 0., 1., -np.inf),
])
def test_ground_plane_rejects_invalid_coefficients(coefficients):
    from flash_chord.scene.collision import CollisionPolicy

    with pytest.raises(ValueError, match='ground_plane'):
        CollisionPolicy(ground_plane=coefficients)


@pytest.mark.parametrize('coefficients', [None, (-.3, .2, 2., .8)])
def test_one_world_scene_uses_exact_measured_or_legacy_floor(monkeypatch, tmp_path, coefficients):
    """Inspect a finalized collision shape, including orientation and plane sign."""
    import newton
    import warp as wp
    from scipy.spatial.transform import Rotation
    from flash_chord.assets import SHARPAWAVE_URDF_DIR
    from flash_chord.embodiments.binding import build_robot_reference_binding
    from flash_chord.embodiments.sharpa_hands import SharpaHands, SharpaHandsConfig
    from flash_chord.scene import builder as scene_builder
    from flash_chord.scene.collision import CollisionPolicy

    # The floor contract needs no external demonstration or object reconstruction.
    # Keep the real robot/FK, replication, finalization and ground collision path.
    reference = SimpleNamespace(num_frames=2, fps=20.)
    robot = SharpaHands(SharpaHandsConfig(
        left_urdf_path=SHARPAWAVE_URDF_DIR/'left_sharpawave_primitive.urdf',
        right_urdf_path=SHARPAWAVE_URDF_DIR/'right_sharpawave_primitive.urdf'))
    def bind_initial_pose(builder, layout, reference):
        q = np.tile(np.asarray(builder.joint_q, np.float32), (reference.num_frames, 1))
        target = np.tile(np.asarray(builder.joint_target_pos, np.float32), (reference.num_frames, 1))
        return build_robot_reference_binding(builder, layout, q, target, reference.fps)
    embodiment = SimpleNamespace(build=robot.build, bind_reference=bind_initial_pose)
    monkeypatch.setattr(scene_builder, 'spawn_objects', lambda *args, **kwargs: [])
    support = tmp_path/'support.usda'
    support.write_text('''#usda 1.0
def Cylinder "table" {
    double radius = 0.3
    double height = 0.04
    double3 xformOp:translate = (0.1, 0.2, 0.3)
}
''')
    policy = CollisionPolicy(ground_plane=coefficients)
    with wp.ScopedDevice('cpu'):
        scene = scene_builder.build_scene(embodiment, reference, world_count=1,
                                         collision=policy, contact_friction=.37, support_usda=support)

    model = scene.model
    plane_ids = np.flatnonzero(model.shape_type.numpy() == int(newton.GeoType.PLANE))
    assert plane_ids.shape == (1,), 'Measured floor must replace the default plane, not add another one'
    plane_id = plane_ids[0]
    assert scene.world_count == 1
    assert scene.support.free_q_ids == scene.support.free_dof_ids == ()
    assert model.joint_coord_count == scene.layout.num_joint_q == 58
    assert model.joint_dof_count == scene.layout.num_joint_dof == 56
    assert scene.support.shapes == scene.collision_layout.support
    np.testing.assert_array_equal(model.shape_body.numpy()[scene.support.shapes.start:scene.support.shapes.stop],
                                  scene.support.body_id)
    assert model.shape_count == scene.collision_layout.shape_count+1
    assert model.shape_body.numpy()[plane_id] == -1
    assert model.shape_flags.numpy()[plane_id] & int(newton.ShapeFlags.COLLIDE_SHAPES)
    np.testing.assert_array_equal(model.shape_scale.numpy()[plane_id], [0., 0., 0.])
    assert model.shape_material_mu.numpy()[plane_id] == pytest.approx(.37)

    transform = model.shape_transform.numpy()[plane_id]
    rotation = Rotation.from_quat(transform[3:])
    normal = rotation.apply([0., 0., 1.])
    expected = np.asarray((0., 0., 1., 0.) if coefficients is None else coefficients)
    expected = expected/np.linalg.norm(expected[:3])
    np.testing.assert_allclose(normal, expected[:3], atol=1e-7)
    # Several world points on the actual infinite collision plane must satisfy
    # ax+by+cz+d=0. This catches normal/offset sign and normalization mistakes.
    points = rotation.apply([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])+transform[:3]
    np.testing.assert_allclose(points @ expected[:3]+expected[3], 0., atol=1e-7)
    if coefficients is None:
        assert policy.ground_plane is None
        np.testing.assert_array_equal(transform[:3], [0., 0., 0.])
        np.testing.assert_allclose(points[:, 2], 0., atol=1e-7)
    else:
        np.testing.assert_allclose(policy.ground_plane, expected)
        assert transform[2] < -.3
        assert np.ptp(points[:, 2]) > .1


def test_fixed_support_contact_matches_static_world_without_artificial_dofs(tmp_path):
    """The imported support must solve like the same world-static geometry."""
    import mujoco
    import newton
    import warp as wp
    from flash_chord.scene.support import add_support_surfaces

    path = tmp_path/'support.usda'
    path.write_text('''#usda 1.0
def Cylinder "table" {
    double radius = 0.3
    double height = 0.04
    double3 xformOp:translate = (0, 0, 0.18)
}
''')
    trajectories = []
    with wp.ScopedDevice('cpu'):
        for imported in (False, True):
            builder = newton.ModelBuilder()
            newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
            box = builder.add_link(mass=.2, label='box')
            joint = builder.add_joint_free(child=box, label='box_free')
            builder.add_articulation([joint], label='box')
            builder.add_shape_box(box, hx=.05, hy=.05, hz=.05)
            builder.joint_q[:3] = [0., 0., .255]
            box_q = np.asarray(builder.joint_q).copy()
            if imported:
                binding = add_support_surfaces(builder, path)
                assert binding.free_q_ids == binding.free_dof_ids == ()
                np.testing.assert_array_equal(builder.joint_q, box_q)
            else:
                builder.add_shape_cylinder(body=-1, radius=.3, half_height=.02,
                    xform=wp.transform(wp.vec3(0., 0., .18), wp.quat_identity()))
            model = builder.finalize(device='cpu')
            assert model.joint_coord_count == 7 and model.joint_dof_count == 6
            solver = newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=True,
                use_mujoco_contacts=True, solver='newton', integrator='implicitfast',
                iterations=100, ls_iterations=50, njmax=32, nconmax=16)
            backend, state = solver.mj_model, solver.mj_data
            assert backend.nq == 7 and backend.nv == 6
            assert np.max(backend.dof_armature) < 1., 'Support must not add enormous artificial armature'
            initial = model.state()
            initial.joint_q.assign(model.joint_q)
            newton.eval_fk(model, initial.joint_q, initial.joint_qd, initial)
            solver._update_mjc_data(state, model, initial)
            mujoco.mj_forward(backend, state)
            assert state.qpos[2] == pytest.approx(.255)
            backend.opt.timestep = .0025
            trajectory = []
            for _ in range(400):
                mujoco.mj_step(backend, state)
                trajectory.append(np.r_[state.qpos[:3], state.qvel])
            trajectory = np.asarray(trajectory)
            assert np.isfinite(trajectory).all()
            assert .24 < trajectory[-1, 2] < .26
            assert trajectory[:, 2].max() < .27
            assert np.max(np.linalg.norm(trajectory[:, :2], axis=1)) < .1
            trajectories.append(trajectory)
    np.testing.assert_allclose(trajectories[1], trajectories[0], atol=2e-6, rtol=1e-5)
