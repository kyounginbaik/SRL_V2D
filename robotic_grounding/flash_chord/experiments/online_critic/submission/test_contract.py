"""Submission guards reject incomplete/invalid evidence instead of padding it."""

from pathlib import Path
import os
import sys
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from contract import validate_trajectory
from pack_submission import inspect_results, pack
from flash_chord.evaluation.submission import write_object_trajectories_parquet


ROSTER = {"1": dict(object_names=["pot", "lid"], world_indices=[0], frame_indices=[0, 1], row_count=4)}


def make_rollout(path):
    pose = np.array([1., 2., 3., 0., 0., 0., 1.])
    rollout = SimpleNamespace(
        achieved_pose_w=np.broadcast_to(pose, (2, 1, 2, 7)).copy(),
        reference_pose_w=np.broadcast_to(pose, (2, 2, 7)).copy(),
        object_body_names=("pot", "lid"), body_object_ids=np.array([0, 1]),
        non_finite_worlds=(), start_frame=0, control_fps=20,
    )
    write_object_trajectories_parquet(path, rollout, checkpoint_sha256="a"*64,
        eval_script_sha256="b"*64, eval_code_sha256="c"*64,
        reference_parquet="/frozen/episode_000001/reference.parquet")
    return path


def change_column(path, name, values):
    table = pq.read_table(path)
    table = table.set_column(table.schema.get_field_index(name), name, pa.array(values))
    pq.write_table(table, path)


def test_valid_frozen_rollout(tmp_path):
    path = make_rollout(tmp_path / "episode_000001.parquet")
    report = validate_trajectory(path, episode="episode_000001", checkpoint_sha256="a"*64, roster=ROSTER)
    assert report["valid"] and report["rows"] == 4 and report["frames"] == 2


@pytest.mark.parametrize("change,error", [
    (lambda p: pq.write_table(pq.read_table(p).slice(0, 3), p), "Row roster mismatch"),
    (lambda p: change_column(p, "reference_frame", [0, 0, 0, 0]), "Duplicate"),
    (lambda p: change_column(p, "reference_frame", [0, 0, 2, 2]), "Row roster mismatch"),
    (lambda p: change_column(p, "object_body_name", ["lid", "pot", "lid", "pot"]), "Object slot"),
    (lambda p: change_column(p, "environment_valid", [True, True, False, True]), "invalid simulation"),
    (lambda p: change_column(p, "achieved_position_x", [float("nan"), 0., 0., 0.]), "Non-finite"),
    (lambda p: change_column(p, "achieved_quaternion_w", [0., 0., 0., 0.]), "Non-unit"),
    (lambda p: change_column(p, "reference_frame", [0., 0., 1., 1.]), "must be integers"),
])
def test_invalid_evidence_is_rejected(tmp_path, change, error):
    path = make_rollout(tmp_path / "episode_000001.parquet")
    change(path)
    with pytest.raises(ValueError, match=error):
        validate_trajectory(path, roster=ROSTER)


def test_wrong_checkpoint_or_episode_is_rejected(tmp_path):
    path = make_rollout(tmp_path / "episode_000001.parquet")
    with pytest.raises(ValueError, match="checkpoint hash"):
        validate_trajectory(path, checkpoint_sha256="d"*64, roster=ROSTER)
    with pytest.raises(ValueError, match="different episode"):
        validate_trajectory(path, episode=3, roster=ROSTER)


def test_invalid_provenance_is_rejected(tmp_path):
    path = make_rollout(tmp_path / "episode_000001.parquet")
    table = pq.read_table(path)
    metadata = dict(table.schema.metadata)
    metadata[b"flash_chord.checkpoint_sha256"] = b"unknown"
    pq.write_table(table.replace_schema_metadata(metadata), path)
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        validate_trajectory(path, roster=ROSTER)


def test_valid_full_result_supersedes_first_pass(tmp_path):
    full, smoke = tmp_path / "full", tmp_path / "smoke"
    make_rollout(smoke / "episode_000001.parquet")
    report = inspect_results(full, fallback=smoke, roster=ROSTER)
    assert report["selected"]["1"]["fallback"] and not report["missing_episodes"]
    make_rollout(full / "episode_000001.parquet")
    report = inspect_results(full, fallback=smoke, roster=ROSTER)
    assert not report["selected"]["1"]["fallback"]
    change_column(full / "episode_000001.parquet", "environment_valid", [False]*4)
    report = inspect_results(full, fallback=smoke, roster=ROSTER)
    assert report["selected"]["1"]["fallback"] and report["invalid_inputs"]["1"]


def test_missing_episodes_never_produce_submission(tmp_path):
    output = tmp_path / "submission.csv"
    code = pack(tmp_path / "empty", output, kit=None, code_commit="unknown", allow_incomplete=True)
    assert code == 0 and not output.exists()
    import json
    report = json.loads(output.with_suffix(".readiness.json").read_text())
    assert report["ready"] is False and len(report["missing_episodes"]) == 20


def test_export_uses_frozen_unassisted_full_horizon_and_preserves_valid_output(tmp_path, monkeypatch):
    import export_episode
    from contract import digest

    checkpoint = tmp_path / "policy.safetensors"
    checkpoint.write_bytes(b"synthetic test checkpoint")
    output = tmp_path / "episode_000001.parquet"
    output.write_bytes(b"previous validated result")
    observed = {}

    def evaluate(path, **kwargs):
        observed.update(kwargs)
        make_rollout(kwargs["object_trajectories_output"])
        return SimpleNamespace(environment_steps=1048576)

    monkeypatch.setitem(sys.modules, "flash_chord.evaluation.harness", SimpleNamespace(
        evaluate_checkpoint=evaluate, write_evaluation_report=lambda *args: None))
    # Wrong checkpoint digest in this synthetic rollout must prevent publication.
    with pytest.raises(ValueError, match="Row roster mismatch"):
        export_episode.export(checkpoint, 1, output)
    assert output.read_bytes() == b"previous validated result"
    assert observed["world_count"] == 1 and observed["start_frame"] == 0
    assert observed["reset_mode"] == "explicit"
    assert observed["motion_start_frame"] == 0 and observed["motion_end_frame"] == -1
    assert observed["eval_script"].name == "export_episode.py"


@pytest.mark.skipif(not os.environ.get("V2D_SUBMISSION_KIT"), reason="Official participant kit not mounted")
def test_official_packer_accepts_complete_public_roster_fixture(tmp_path):
    """Synthetic test poses stay inside pytest's temporary directory."""
    from contract import load_roster
    import json

    results = tmp_path / "synthetic_fixture"
    for key, item in load_roster().items():
        frames, bodies = len(item["frame_indices"]), len(item["object_names"])
        pose = np.array([0., 0., 0., 0., 0., 0., 1.])
        rollout = SimpleNamespace(
            achieved_pose_w=np.broadcast_to(pose, (frames, 1, bodies, 7)).copy(),
            reference_pose_w=np.broadcast_to(pose, (frames, bodies, 7)).copy(),
            object_body_names=tuple(item["object_names"]), body_object_ids=np.arange(bodies),
            non_finite_worlds=(), start_frame=0, control_fps=20,
        )
        write_object_trajectories_parquet(results / f"episode_{int(key):06d}.parquet", rollout,
            checkpoint_sha256="a"*64, eval_script_sha256="b"*64, eval_code_sha256="c"*64,
            reference_parquet=f"/synthetic_test_fixture/episode_{int(key):06d}/reference.parquet")
    output = tmp_path / "synthetic_test_submission.csv"
    assert pack(results, output, kit=os.environ["V2D_SUBMISSION_KIT"],
                code_commit="https://github.com/synthetic/fixture/commit/" + "d"*40) == 0
    report = json.loads(output.with_suffix(".readiness.json").read_text())
    assert report["ready"] and report["available_episodes"] == 20
    assert not report["missing_episodes"]
