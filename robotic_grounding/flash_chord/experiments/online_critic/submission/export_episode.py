#!/usr/bin/env python3
"""Evaluate one frozen trained policy and publish an exact Track 3 trajectory."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

from contract import atomic_json, digest, episode_index, validate_kit_roster, validate_trajectory


def export(checkpoint, episode, output, *, kit=None, selection_criterion="explicit_checkpoint"):
    # Imports are deferred so roster validation and packing do not require a GPU.
    from flash_chord.evaluation.harness import evaluate_checkpoint, write_evaluation_report

    checkpoint = Path(checkpoint).resolve()
    output = Path(output).resolve()
    ep = episode_index(episode)
    if output.name != f"episode_{ep:06d}.parquet":
        raise ValueError("Use the exact episode_<six digits>.parquet output name")
    if kit:
        validate_kit_roster(kit)
    checkpoint_hash = digest(checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path = output.with_suffix(".validation.json")
    # Keep interrupted or invalid evaluations out of the packer's input directory.
    with tempfile.TemporaryDirectory(prefix=".export-", dir=output.parent) as temporary:
        staged = Path(temporary) / output.name
        try:
            outcome = evaluate_checkpoint(
                checkpoint, world_count=1, start_frame=0, reset_mode="explicit",
                motion_start_frame=0, motion_end_frame=-1,
                object_trajectories_output=staged, eval_script=Path(__file__),
            )
            if digest(checkpoint) != checkpoint_hash:
                raise ValueError("Checkpoint changed during evaluation")
            report = validate_trajectory(staged, episode=ep, checkpoint_sha256=checkpoint_hash)
            report.update(checkpoint=str(checkpoint), output=str(output),
                          unassisted=True, voc_scale=0.0, full_horizon=True,
                          frozen_policy=True, reset_mode="explicit",
                          checkpoint_selection_criterion=selection_criterion,
                          selection_uses_reconstructed_reference_not_challenge_ground_truth=True,
                          training_environment_steps=int(outcome.environment_steps),
                          success_not_required_for_schema_validity=True)
            metrics = Path(temporary) / "metrics.json"
            write_evaluation_report(metrics, outcome)
            os.replace(staged, output)
            os.replace(metrics, output.with_suffix(".metrics.json"))
            atomic_json(report_path, report)
            return report
        except Exception as error:
            atomic_json(report_path, dict(valid=False, episode_index=ep,
                        checkpoint=str(checkpoint), error=str(error),
                        previous_output_preserved=output.is_file()))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--kit", type=Path, default=os.environ.get("V2D_SUBMISSION_KIT"))
    parser.add_argument("--selection-criterion", default="explicit_checkpoint",
                        choices=("explicit_checkpoint", "archived_best_prefix_then_cost", "final_checkpoint"))
    args = parser.parse_args()
    report = export(args.checkpoint, args.episode, args.output, kit=args.kit,
                    selection_criterion=args.selection_criterion)
    print(f"Validated {args.output}: {report['frames']} frames, {report['rows']} rows", flush=True)


if __name__ == "__main__":
    main()
