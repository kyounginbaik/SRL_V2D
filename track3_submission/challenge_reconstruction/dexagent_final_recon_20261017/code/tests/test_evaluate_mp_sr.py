"""Meaningful metric-contract tests; no simulator/GPU required."""
import ast
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluate_mp_sr as metric


def sample(names=("white_pot", "white_pot_lid"), frames=21, worlds=1):
    q = np.zeros((frames, worlds, len(names), 4)); q[..., 0] = 1
    p = np.zeros((frames, worlds, len(names), 3))
    p[..., 0] = np.arange(len(names)) * .25
    return dict(timestamps_seconds=np.arange(frames) / 20., object_positions=p,
                object_quaternions_wxyz=q, object_names=list(names))


def geometry(names):
    return dict(objects={name: dict(anchor_local_m=[0., 0., 0.],
                                   symmetry=metric.SYMMETRIES.get(name, "none"),
                                   axis_local=[0., 0., 1.] if name in metric.SYMMETRIES else [0., 0., 0.])
                         for name in names})


def change_rotation(data, body, euler, degrees):
    data["object_quaternions_wxyz"][:, :, body] = Rotation.from_euler(euler, degrees, degrees=True).as_quat()[[3, 0, 1, 2]]


def score(ref, got):
    return metric.evaluate_arrays(ref, got, geometry(ref["object_names"]))


def test_identity_passes_and_does_not_mutate_inputs():
    ref, got = sample(), sample()
    before = got["object_positions"].copy()
    result = score(ref, got)
    assert result["reference_proxy_mp_sr"] == 1
    assert result["declared_frame_mp_sr"] == 1
    assert result["full_horizon"] and not result["leaderboard_score"]
    np.testing.assert_array_equal(before, got["object_positions"])


def test_all_objects_must_pass_strict_threshold_and_mean_not_max():
    ref, got = sample(), sample()
    got["object_positions"][:, :, 1, 2] += .03
    assert score(ref, got)["reference_proxy_mp_sr"] == 0
    got["object_positions"][:, :, 1, 2] = 0
    got["object_positions"][10, :, 1, 2] = .2
    assert score(ref, got)["reference_proxy_mp_sr"] == 1  # 20cm/21 is less than 3cm
    got["object_positions"][:, :, 1, 2] = .031
    assert score(ref, got)["reference_proxy_mp_sr"] == 0


def test_worlds_average_after_every_object_gate():
    ref, got = sample(), sample(worlds=2)
    got["object_positions"][:, 1, 1, 2] = .031
    assert score(ref, got)["reference_proxy_mp_sr"] == .5


@pytest.mark.parametrize("name,angle,expected", [("white_pot_lid", 180, 1), ("white_pot_lid", 90, 0),
                                               ("wooden_spoon", 180, 0), ("beige_cup", 72, 1), ("blue_cup", 139, 1)])
def test_only_declared_yaw_symmetry_is_ignored(name, angle, expected):
    ref, got = sample(("white_pot", name)), sample(("white_pot", name))
    change_rotation(got, 1, "z", angle)
    assert score(ref, got)["reference_proxy_mp_sr"] == expected


@pytest.mark.parametrize("tilt,expected", [(29, 1), (31, 0), (180, 0)])
def test_cup_tilt_and_upside_down_are_scored(tilt, expected):
    ref, got = sample(("white_pot", "beige_cup")), sample(("white_pot", "beige_cup"))
    change_rotation(got, 1, "x", tilt)
    assert score(ref, got)["reference_proxy_mp_sr"] == expected


def test_quaternion_sign_is_invariant():
    ref, got = sample(), sample()
    got["object_quaternions_wxyz"] *= -1
    assert score(ref, got)["reference_proxy_mp_sr"] == 1


def test_incomplete_horizon_rejected_instead_of_prefix_success():
    ref, got = sample(), sample()
    for key in ("timestamps_seconds", "object_positions", "object_quaternions_wxyz"):
        got[key] = got[key][:-1]
    with pytest.raises(ValueError, match="Incomplete horizon"):
        score(ref, got)


def test_missing_middle_frames_and_missing_object_rejected():
    ref, got = sample(), sample()
    for key in ("timestamps_seconds", "object_positions", "object_quaternions_wxyz"):
        got[key] = np.delete(got[key], [5, 6], axis=0)
    with pytest.raises(ValueError, match="Missing temporal coverage"):
        score(ref, got)
    with pytest.raises(ValueError, match="every reference object"):
        score(ref, sample(("white_pot",)))


def test_nonfinite_and_zero_quaternion_rejected():
    ref, got = sample(), sample()
    got["object_positions"][3, 0, 1, 0] = np.nan
    with pytest.raises(ValueError, match="Non-finite"):
        score(ref, got)
    got = sample(); got["object_quaternions_wxyz"][5] = 0
    with pytest.raises(ValueError, match="quaternion"):
        score(ref, got)


def test_interpolation_uses_absolute_time_without_duration_warp():
    ref, got = sample(), sample(frames=101)
    got["timestamps_seconds"] /= 5  # 100Hz, still one second
    ref["object_positions"][:, :, 1, 2] = ref["timestamps_seconds"][:, None] * .1
    got["object_positions"][:, :, 1, 2] = got["timestamps_seconds"][:, None] * .1
    assert score(ref, got)["reference_proxy_mp_sr"] == 1
    got["timestamps_seconds"] *= .5
    with pytest.raises(ValueError, match="Incomplete horizon"):
        score(ref, got)


def test_one_scene_alignment_keeps_relative_geometry_error():
    ref, got = sample(), sample()
    got["object_positions"] += [.5, .1, .2]
    result = score(ref, got)
    assert result["reference_proxy_mp_sr"] == 1
    assert result["declared_frame_mp_sr"] == 0
    got["object_positions"][:, :, 1, 2] += .04
    assert score(ref, got)["reference_proxy_mp_sr"] == 0


def test_explicit_adapter_transform_restores_same_frame_without_fitting():
    ref, got = sample(), sample()
    t = np.eye(4); t[:3, :3] = Rotation.from_euler("xyz", [20, 35, 40], degrees=True).as_matrix(); t[:3, 3] = [.2, -.1, .4]
    moved = metric.apply_world_transform(metric.validate_poses(got), t)
    result = metric.evaluate_arrays(ref, moved, geometry(ref["object_names"]), np.linalg.inv(t))
    assert result["declared_frame_mp_sr"] == 1
    with pytest.raises(ValueError, match="rigid"):
        metric.apply_world_transform(got, np.diag([2., 2., 2., 1.]))


def test_no_invented_symmetries_or_unreviewed_axis():
    ref, got = sample(), sample()
    g = geometry(ref["object_names"]); g["objects"]["white_pot"]["symmetry"] = "continuous_axial"
    with pytest.raises(ValueError, match="Undeclared object symmetry"):
        metric.evaluate_arrays(ref, got, g)
    g = geometry(ref["object_names"]); g["objects"]["white_pot_lid"]["axis_local"] = [0., 0., 0.]
    with pytest.raises(ValueError, match="unit axis"):
        metric.evaluate_arrays(ref, got, g)


def test_parity_with_downloaded_official_metric_functions():
    directory = Path(__file__).resolve().parents[2] / "metric_sources/official/v2d_submission_kit/v2dlb"
    if not directory.exists():
        pytest.skip("Official downloaded source is optional for portable unit tests")
    namespace = dict(np=np)
    wanted = {"_rotation_matrix", "_quaternion_multiply", "_quaternion_geodesic_angle", "_axis_basis",
              "_align_registered_scene", "_symmetry_orientation_errors", "_registered_pose_metric"}
    for filename in ("object_pose.py", "track3_pose.py"):
        tree = ast.parse((directory / filename).read_text())
        module = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted], type_ignores=[])
        exec(compile(module, str(directory / filename), "exec"), namespace)
    names = ["white_pot", "white_pot_lid", "beige_cup", "blue_cup", "wooden_spoon"]
    rng = np.random.default_rng(52)
    for cup_only in (False, True):
        chosen = names[2:4] if cup_only else names
        ref, got = sample(chosen), sample(chosen, worlds=3)
        for poses in (ref, got):
            positions = poses["object_positions"]
            positions += rng.normal(0., .02, positions.shape)
            q = Rotation.random(np.prod(positions.shape[:-1]), random_state=rng).as_quat().reshape(*positions.shape[:-1], 4)
            poses["object_quaternions_wxyz"] = q[..., [3, 0, 1, 2]]
        g = geometry(chosen)
        _, axes, kinds = metric.validate_geometry(chosen, g)
        def official_poses(data):
            return np.concatenate([data["object_positions"], data["object_quaternions_wxyz"][..., [1, 2, 3, 0]]], axis=-1)
        orders = [2 if kind == "half_turn" else 360 if kind == "continuous_axial" else 1 for kind in kinds]
        a, r = official_poses(got), official_poses(ref)
        np.testing.assert_allclose(metric.angular_errors(got["object_quaternions_wxyz"], ref["object_quaternions_wxyz"], axes, kinds),
                                   namespace["_symmetry_orientation_errors"](a, r, axes, orders), atol=1e-7)
        aligned, _ = metric.align_initial_scene(got, ref, axes, kinds)
        aligned_official = namespace["_align_registered_scene"](a, r, axes)
        np.testing.assert_allclose(aligned["object_positions"], aligned_official[..., :3], atol=1e-10)
        np.testing.assert_allclose(metric.rotation_matrices(aligned["object_quaternions_wxyz"]),
                                   namespace["_rotation_matrix"](aligned_official[..., 3:]), atol=1e-10)
        expected = namespace["_registered_pose_metric"]("maniptrans_sr", aligned_official, r, [], axes, orders)
        assert metric.score_aligned(aligned, ref, axes, kinds)["mp_sr"] == expected
