"""Submission trajectory export stays readable by the challenge kit."""

from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq

from flash_chord.evaluation.submission import write_object_trajectories_parquet


def test_write_object_trajectories_parquet(tmp_path):
    pose = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0])
    achieved = np.broadcast_to(pose, (2, 1, 2, 7)).copy()
    reference = np.broadcast_to(pose, (2, 2, 7)).copy()
    rollout = SimpleNamespace(
        achieved_pose_w=achieved,
        reference_pose_w=reference,
        object_body_names=("white_pot", "white_pot_lid"),
        body_object_ids=np.array([0, 1]),
        non_finite_worlds=(),
        start_frame=5,
        control_fps=50,
    )
    destination = tmp_path / "episode_000001.parquet"
    write_object_trajectories_parquet(
        destination,
        rollout,
        checkpoint_sha256="checkpoint-hash",
        eval_script_sha256="script-hash",
        eval_code_sha256="code-hash",
        reference_parquet="/v2d/assets/sequence_id=episode_000001/data.parquet",
    )

    table = pq.read_table(destination)
    metadata = {key.decode(): value.decode() for key, value in table.schema.metadata.items()}
    assert table.num_rows == 4
    assert table["reference_frame"].to_pylist() == [5, 5, 6, 6]
    assert table["object_body_name"].to_pylist() == [
        "white_pot", "white_pot_lid", "white_pot", "white_pot_lid"
    ]
    assert table["achieved_quaternion_w"].to_pylist() == [1.0] * 4
    assert metadata["flash_chord.episode_index"] == "1"
    assert metadata["flash_chord.schema"] == "flash_chord_object_trajectories_v1"
