from types import SimpleNamespace as NS
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from flash_chord.retargeting.reference_relabel import ReferenceRelabel


def fixture():
    layout = NS(num_hands=2, total_arms=0, total_fingers=44, wrist_pos_start=0,
                wrist_quat_start=6, command_wrist_start=14, command_finger_start=28)
    scale = NS(scale=1.)
    config = NS(wrist_position=scale, wrist_orientation=scale, command_wrist=scale,
                command_finger=scale, make_wrist_quat_unique=True)
    values = np.zeros((3, 58), np.float32)
    values[:, 3] = 1; values[:, 10] = 1
    jacobian = np.zeros((3, 58, 56), np.float32)
    jacobian[1, 0, 0] = .02; jacobian[1, 14, 6] = .2
    obs = np.zeros((3, 80), np.float32)
    # Rotated wrist: a world-x reference displacement appears along local -y.
    q = Rotation.from_euler('z', 90, degrees=True).as_quat(scalar_first=True)
    obs[:, 6:10] = q; obs[:, 10] = 1
    obs[:, 17:21] = q*np.array([1, -1, -1, -1]); obs[:, 24] = 1
    obs[:, 72:] = np.arange(8)  # sentinel object/contact/past-action values
    return ReferenceRelabel(values, jacobian, layout, config), jnp.asarray(obs)


def test_zero_relabel_matches_and_changes_only_reference_fields():
    relabel, obs = fixture()
    assert relabel.validate_zero(obs, jnp.arange(3)) < 1e-6
    theta = jnp.zeros(56).at[0].set(.5).at[6].set(.25)
    out = np.asarray(relabel(obs, jnp.arange(3), theta))
    np.testing.assert_allclose(out[1, 14:17], [0, -.01, 0], atol=1e-7)
    assert np.isclose(out[1, 28], .05)
    np.testing.assert_allclose(out[[0, 2]], np.asarray(obs)[[0, 2]], atol=1e-7)
    np.testing.assert_array_equal(out[:, 72:], obs[:, 72:])
    np.testing.assert_array_equal(out[:, :14], obs[:, :14])


def test_relabel_derivative_matches_independent_rotation_and_finger_change():
    relabel, obs = fixture()
    derivative = jax.jacfwd(lambda t: relabel(obs, jnp.arange(3), t))(jnp.zeros(56))
    np.testing.assert_allclose(derivative[1, 14:17, 0], [0, -.02, 0], atol=1e-7)
    assert np.isclose(derivative[1, 28, 6], .2)
    assert np.isfinite(np.asarray(derivative)).all()


@pytest.mark.parametrize('wrists_only', [False, True])
def test_native_sharpa_fk_linearization_matches_a_fresh_bounded_edit(wrists_only, monkeypatch, tmp_path):
    import newton
    import warp as wp
    from flash_chord.assets import SHARPAWAVE_URDF_DIR
    from flash_chord.embodiments.sharpa_hands import SharpaHands, SharpaHandsConfig
    from flash_chord.embodiments.binding import build_robot_reference_binding
    from flash_chord.retargeting.local_correction import local_candidate
    from flash_chord.retargeting.reference_relabel import binding_for_q, reference_values
    with wp.ScopedDevice('cpu'):
        embodiment = SharpaHands(SharpaHandsConfig(
            left_urdf_path=SHARPAWAVE_URDF_DIR/'left_sharpawave_primitive.urdf',
            right_urdf_path=SHARPAWAVE_URDF_DIR/'right_sharpawave_primitive.urdf'))
        builder = newton.ModelBuilder()
        layout = embodiment.build(builder)
        model = builder.finalize(device='cpu', skip_all_validations=True)
        qids = np.asarray(layout.scalar_joints.q_ids); dids = np.asarray(layout.scalar_joints.dof_ids)
        balls = [(builder.joint_q_start[i], builder.joint_qd_start[i])
                 for i, kind in enumerate(builder.joint_type) if kind == newton.JointType.BALL]
        q = np.tile(np.asarray(builder.joint_q, np.float32), (3, 1))
        lower = model.joint_limit_lower.numpy()[dids]; upper = model.joint_limit_upper.numpy()[dids]
        q[:, qids] = np.clip(q[:, qids], lower, upper)
        target = np.tile(np.asarray(builder.joint_target_pos, np.float32), (3, 1))
        target[:, dids] = q[:, qids]
        binding = build_robot_reference_binding(builder, layout, q, target, 20.)
        projector = NS(builder=builder, layout=layout, qids=qids, dids=dids, ball_joints=balls,
                       lower=lower, upper=upper)
        rows = [np.eye(layout.num_joint_q)[i] for i in qids]
        rows += [np.eye(layout.num_joint_q)[i+a] for i, _ in balls for a in range(3)]
        wrist_ids = {qid for name, qid in zip(layout.scalar_joints.names, qids)
                     if name.endswith(('wrist_x', 'wrist_y', 'wrist_z'))}
        wrist_ids.update(i+a for i, _ in balls for a in range(3))
        if wrists_only:
            rows = [row for row in rows if set(np.flatnonzero(row)).issubset(wrist_ids)]
        dimensions = 12 if wrists_only else 56
        assert len(rows) == dimensions
        editor = NS(projector=projector, baseline=binding, binding=binding, joint_basis=np.asarray(rows),
                    basis=np.array([[0.], [1.], [0.]]), options={})
        editor.candidate_joint_q = lambda a: local_candidate(editor, a)
        simple, _ = fixture()
        strategy = NS(layout=NS(**vars(simple.layout), sides=('right', 'left')), config=simple.config)
        linear = ReferenceRelabel.from_editor(editor, strategy)
        assert linear.dimensions == dimensions
        theta = np.random.default_rng(3).normal(0, .003, dimensions).astype(np.float32)
        actual = reference_values(binding_for_q(editor, editor.candidate_joint_q(theta)), strategy.layout.sides)
        predicted = np.asarray(linear.reference(jnp.arange(3), theta))
        for hand in range(2):
            ids = slice(7*hand+3, 7*hand+7)
            actual[:, ids] *= np.where(np.sum(actual[:, ids]*predicted[:, ids], axis=-1, keepdims=True) < 0, -1, 1)
        np.testing.assert_allclose(predicted, actual, atol=7e-5, rtol=1e-4)
        np.testing.assert_array_equal(editor.candidate_joint_q(theta)[[0, 2]], q[[0, 2]])
        # Feasible actor output never depends on a subsequently clipped edit.
        feasible = np.asarray(linear.parameterize(jnp.zeros(dimensions), .25))
        editor.candidate_joint_q(feasible)
        np.testing.assert_allclose(editor.last_projection_scales, 1., atol=1e-5)
        assert np.all(feasible >= np.asarray(linear.lower))
        assert np.all(feasible <= np.asarray(linear.upper))
        if wrists_only:
            finger_ids = [i for i in qids if i not in wrist_ids]
            np.testing.assert_array_equal(editor.candidate_joint_q(theta)[:, finger_ids], q[:, finger_ids])
            np.testing.assert_array_equal(actual[:, 14:], np.asarray(linear.values)[:, 14:])
            np.testing.assert_array_equal(linear.jacobian[:, 14:], 0.)
        # Exercise every inherited constructor after the expensive shared scene
        # setup. The parallel base still has several temporal modes here; the
        # local constructor must collapse those before accepting one 12/56D edit.
        import json
        from flash_chord.retargeting.ttsa_editor import TTSAReferenceEditor
        from flash_chord.retargeting.wrench_hypergradient_editor import WrenchHypergradientEditor
        from flash_chord.training.flash_sac.config import TrainingConfig, NetworkConfig
        training = TrainingConfig(world_count=4, network=NetworkConfig(actor_head_mode='residual'))
        setup = NS(scene=NS(world_count=4), reference=NS())
        def scene_setup(instance, **kwargs):
            instance.options = kwargs['options']
            instance.projector = projector
            instance.baseline = instance.binding = binding
            instance.joint_basis = np.asarray(rows)
            instance.basis = np.ones((3, 2))
            instance.blend = .25
            instance.setup = kwargs['setup']
            instance.eval_config = {}
            instance.training = training
            instance.config = {'env': {'reset': {'fixed_initial_state': True, 'voc_decay_steps': 0}}}
            instance.contact_proximity = None
            instance.version = 0
            instance.output = tmp_path/'references'
            instance.output.mkdir()
            (tmp_path/'ttsa_manifest.json').write_text('{}')
        monkeypatch.setattr(TTSAReferenceEditor, '__init__', scene_setup)
        constructed = WrenchHypergradientEditor(setup=setup, options=dict(
            edit_scope='wrist' if wrists_only else 'full', study_variant='chunk',
            eval_worlds=4, upper_trials=4, fixed_start=True, acceptance_mode='local_nominal'))
        assert constructed.upper_action_dim == dimensions
        assert constructed.options['edit_scope'] == ('wrists_only' if wrists_only else 'full')
        assert constructed.basis.shape == (3, 1)
        manifest = json.loads((tmp_path/'ttsa_manifest.json').read_text())
        assert manifest['upper_action_dimension'] == dimensions
        assert manifest['upper_edit_scope'] == constructed.options['edit_scope']
        constructed.candidate_joint_q(np.zeros(dimensions))
