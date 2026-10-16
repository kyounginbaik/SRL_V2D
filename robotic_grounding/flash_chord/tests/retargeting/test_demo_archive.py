"""Archived inference pairs carry the exact curriculum and reference contract."""
from types import SimpleNamespace as NS
import json

import jax.numpy as jnp
import numpy as np
import pytest

from flash_chord.evaluation.harness import _checkpoint_curriculum_stage, _evaluation_curriculum_stage
from flash_chord.lifecycle.curriculum import CurriculumStage
from flash_chord.retargeting.demo_archive import DemoArchive
from flash_chord.retargeting.response_editor import ResponseBilevelEditor
from flash_chord.training.checkpoint import read_checkpoint_metadata
from flash_chord.training.checkpoint_metadata import checkpoint_config


def editor_case(tmp_path):
    editor = ResponseBilevelEditor.__new__(ResponseBilevelEditor)
    editor.output = tmp_path / "references"
    editor.config = {"task": {}, "training": {"curriculum": {"stages": []}}}
    editor.metadata = {}
    editor.projector = NS(layout=None)
    editor.initial_state_sha256 = "fixed-start"
    editor.command_sha256 = "fixed-object-command"
    editor.version = 2
    editor.baseline = NS(fps=20.)
    return editor


def test_archive_saves_actual_adaptive_stage_and_frozen_reference(tmp_path, monkeypatch):
    from flash_chord.retargeting import reference_snapshot

    def save_reference(path, binding, layout, provenance):
        np.savez(path, example=np.zeros(1))
        return "reference-digest"

    monkeypatch.setattr(reference_snapshot, "save_snapshot", save_reference)
    editor = editor_case(tmp_path)
    # A dynamic VOC gate can have more levels than the static curriculum.
    actual = CurriculumStage(.025, {"tracking": 2., "termination": -10.},
                             reset_to_first_frame_probability=1.)
    editor.set_curriculum_metadata(16, actual)
    learner = NS(actor=NS(params={"weight": jnp.ones(2)}, batch_stats={}))
    pair = DemoArchive(editor)._save_pair(None, learner, np.zeros((10,2)), np.zeros((10,2)),
        {}, 1., 100, "baseline", {"current_success_prefix_frames": 6})
    metadata = read_checkpoint_metadata(pair["policy"])
    assert int(metadata["curriculum_stage"]) == 16
    assert float(metadata["voc_scale"]) == .025
    training = NS(curriculum=NS(stages=(CurriculumStage(1., {"tracking": 99.}),)))
    restored = _checkpoint_curriculum_stage(training, metadata, "flash_sac")
    assert restored == actual
    evaluation = _evaluation_curriculum_stage(restored, "explicit")
    assert evaluation.voc_scale == 0.
    assert evaluation.objective_weights == actual.objective_weights
    config = checkpoint_config(metadata)
    assert config["task"]["robot_reference_path"] == pair["reference"]
    assert config["task"]["robot_reference_sha256"] == pair["reference_sha256"]
    # Subsequent training-stage changes cannot change this archived snapshot.
    editor.set_curriculum_metadata(17, CurriculumStage(0., {"tracking": 3.}))
    assert json.loads(read_checkpoint_metadata(pair["policy"])["curriculum_stage_json"])["voc_scale"] == .025


def test_archive_fails_before_writing_without_actual_stage(tmp_path):
    editor = editor_case(tmp_path)
    archive = DemoArchive(editor)
    with pytest.raises(RuntimeError, match="actual curriculum stage"):
        archive._save_pair(None, None, None, None, {}, 1., 100, "baseline", {})
    assert not list(archive.directory.iterdir())


def test_old_valid_stage_metadata_remains_supported_and_missing_stage_fails():
    stage = CurriculumStage(.5, {"tracking": 1.})
    training = NS(curriculum=NS(stages=(stage,)))
    assert _checkpoint_curriculum_stage(training, {"curriculum_stage": "0"}, "flash_sac") == stage
    with pytest.raises(ValueError, match="missing its curriculum stage"):
        _checkpoint_curriculum_stage(training, {}, "flash_sac")
    with pytest.raises(ValueError, match="outside its curriculum"):
        _checkpoint_curriculum_stage(training, {"curriculum_stage": "16"}, "flash_sac")
