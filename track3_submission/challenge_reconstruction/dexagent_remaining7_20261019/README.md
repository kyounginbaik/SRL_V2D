# Remaining seven Track 3 episodes

This work processes episodes **14, 15, 16, 20, 25, 28, and 30** from
`/data1/SRL_V2D/final_recon_ours.zip`, using **HaWoR hand reconstruction** as
requested. It preserves the earlier thirteen episode components, including the
selected episode37 ilchold controller.

Read [status.json](status.json) for completed portable components. A successful
search or video alone does not make a component ready: it must also have a
learned checkpoint, a fresh physical rollout, recorded robot actions, independent
control replay, and exact portable/common-evaluator reproduction. Full20
assembly remains pending until every component has passed those checks.

## Inputs and physical refinement

- `source/final_recon_ours/`: extracted user ZIP; `input_manifest.json` records its
  original hash and files.
- `hawor_stereo_recon_v2/`: decoded HaWoR hands, aligned to the supplied camera and
  world using calibrated stereo depth. HaWoR local finger articulation is
  preserved during this reconstruction alignment. See
  `review/hawor_final_validation.json` and the per-episode hand reports.
- `collision_recon/`: the same visual meshes and object targets, with collision
  decomposition repaired where coarse convex pieces filled openings. Episode25
  has an additional audited narrow-slot variant in `collision_recon_slots_v3/`.
- `physics_collision_v2/episodes/`: floating Sharpa scenes and robot references.
  `physics_slots_v3/` contains episode25's additional scene variant.
- `diagnostics/`: actual physical grasp, transport, placement, and release trials.
  Failed trials are preserved. Robot commands can change during refinement;
  object target trajectories, timestamps, and supplied visual geometry remain
  unchanged. A measured robot grasp from an earlier successful episode may seed
  the search; the human-hand reconstruction remains HaWoR.

Physical demonstrations use the repository's DexAgent floating-hand tools and
MuJoCo backend. A trained radial-basis actor takes continuous phase and measured
robot joint positions/velocities and outputs56 bounded actuator position
commands. Each selected actor is evaluated again in fresh closed-loop physics.
The actor fit is not itself evidence of task success; its physical replay is.

## Ready components and full submission

Completed episodes are under `components/episode_XXXXXX`, with actual robot
actions, checkpoints, provenance, and reviewed videos. Their frozen common
evaluations are under `common_components/`; self-contained replay archives are
under `portable_archives/`.

The common evaluator uses the same frozen runtime build as the earlier13.
Preparation and search helpers can differ, while the executed evaluation source
and corresponding manifest hashes must match that build exactly.

See [package_code/README.md](package_code/README.md) for staging and full20
assembly commands. The final payload uses the current official combined pose
and mesh Parquet schema, covering all9,120 required pose rows. Packaging performs
local preflight and uploader validation without uploading to Kaggle.

All local MP-SR results use the **reconstructed object trajectory as a proxy**
for ground truth: every object must have mean position error below3cm and the
official symmetry-aware mean orientation error below30degrees. They are not
official hidden-ground-truth leaderboard scores. End-of-episode release and
contact/video checks are additional evidence beyond this mean-error criterion.

## Reproduce a refinement

Use the established Python environment:

```bash
TASK_PY=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python
TASK_BASE=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019
"$TASK_PY" "$TASK_BASE/code/update_status.py"
```

Per-trial `spec.json`, `settings.json`, `result.json`, and actor training reports
record the actual inputs and choices. Search code is in `code/`; physical trial
and batch logs are retained in `diagnostics/` and `slurm/`. Heavy CPU searches
should run through Slurm rather than the four-core interactive agent allocation.
For replay on another server, use the selected portable archive and its included
reproduction instructions; the original reconstruction models are not required
to replay the saved robot controller.

## Resume on 2026-10-19 (after the Codex session stopped)

- Episode 30 was staged from the already selected actor
  (`physics_collision_v2/sweeper_transfer/liftclear_0.06_s2.5_v1/release_neutral/trial1/actor_grid`).
- Episode 20 was finished by releasing the cup (`diagnostics/episode20_hold/final_cup_release/trial1`)
  and then the pitcher with a neutral opening at 8.0 s
  (`final_pitcher_release_75_neutral/trial1`), training the actor grid on that
  trajectory and staging it. Both hands are clear at the end.
- Episode 25 never passed the proxy metric. The best open-loop rollout is
  `physics_slots_v3/refinement/episode_000025/push_forecast_v4` (3.4 cm / 3.9 cm).
  Measured-error wrist feedback (`code/episode25_feedback.py`) broke the pinch at every
  gain. Actors were fit with `code/best_effort_actor_grid.py` (seeds 44-47); the
  selected one is exported by `package_code/best_effort_component.py` to
  `components_best_effort/episode_000025_push_forecast_v4_s45` and is explicitly
  flagged as best-effort (no portable archive, not dispatcher-executed).
- Full twenty-episode payload: `package_code/assemble_twenty_best_effort.py` via
  `code/assemble_full20.sbatch` wrote
  `/data1/SRL_V2D/challenge_submissions/track3_full_20261019/all_20_episodes`
  (`track3_submission.parquet`, 9,120 rows; official packer, offline preflight and
  uploader local validation passed). Not uploaded.
