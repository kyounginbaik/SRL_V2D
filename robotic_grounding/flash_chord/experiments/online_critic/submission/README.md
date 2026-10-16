These helpers export **measured unassisted simulation trajectories** from frozen trained policies. The evaluator disables assistance and auto-reset, starts from frame zero, and records the complete reference horizon in one world. A valid file can contain task failures: submission validity does not imply manipulation success.

Run inside the training image with the same prepared run and checkpoint assets mounted:

```bash
python submission/export_episode.py \
  --checkpoint /path/to/run/training/policy_1048576.safetensors \
  --episode episode_000001 \
  --output /path/to/campaign/submissions/smoke/episode_000001.parquet
```

Use a separate `pilot` directory for full training results. The exporter validates exact episode, frame, world and object order, finite poses, unit quaternions, and checkpoint/evaluator provenance before publishing the parquet atomically. It also writes metrics and a validation report. Invalid or partial trajectories are never padded. An optional `--kit` path (or `V2D_SUBMISSION_KIT`) checks that the installed official participant kit matches the frozen public roster.

The campaign selects `training/best_demo.json`'s archived policy/reference pair by unassisted successful prefix, then physical cost. Pass `--selection-criterion archived_best_prefix_then_cost` when using that policy. This is selection against reconstructed reference diagnostics, not a claim of the best held-out challenge score. An archived policy retains its own matching robot reference in checkpoint metadata.

Package all 20 episodes using the official kit:

```bash
python submission/pack_submission.py \
  --results /path/to/campaign/submissions/pilot \
  --fallback-results /path/to/campaign/submissions/smoke \
  --kit /path/to/v2d_submission_kit \
  --code-commit https://github.com/OWNER/REPO/commit/FULL_40_CHARACTER_COMMIT_SHA \
  --output /path/to/campaign/submissions/track3_poses.csv \
  --allow-incomplete
```

`--fallback-results` is optional. A valid full result takes priority over its first-pass result. Readiness reports identify every selected checkpoint, any fallback, invalid input and missing episode. `--allow-incomplete` lets monitoring continue before all jobs finish; it does **not** emit a partial CSV. `ready: true` is written only after the official packer succeeds. Packaging creates local files; it does not upload a challenge submission.

`public_roster.json` contains only participant-visible episode IDs, object order, world IDs and frame indices from the official kit. It includes source hashes and contains no held-out poses or labels. Preserve the separate approved mesh submission: these helpers package the policy pose results.
