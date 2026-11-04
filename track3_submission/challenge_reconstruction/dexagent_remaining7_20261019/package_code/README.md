# Full Track 3 assembly

`assemble_twenty.py` is a packaging step. It does not change physics, fabricate
trajectories, retrain policies, alter evaluator hashes, or contact Kaggle.

The existing thirteen-episode mixed package is preserved. Its producing evaluator
build can evaluate the seven additional episodes without any code change, provided
their portable `code/` and `third_party/DexAgent/` trees exactly match the frozen
legacy backend. Training and scene preparation may use new code; the actual
evaluation must use the frozen runtime identified in `build_manifest.json`.

If a replay runtime change becomes necessary, stop using this integration path:
freeze a new build and freshly execute all twenty episodes under it. Do not
replace provenance on already recorded trajectories.

The assembler requires one actual completed common evaluation and one portable
archive for each episode 14, 15, 16, 20, 25, 28 and 30. It verifies the archive's
actual policy bytes, preserved source-reference hash, action recording, meshes,
trajectory grid and evaluator provenance before writing a new release directory.

For each passed learned actor, first run the staging wrapper. It reads the
selected actor from `refinement/episode_XXXXXX/completed.json` by default:

```bash
TASK_PY=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python
TASK_BASE=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019
"$TASK_PY" "$TASK_BASE/package_code/stage_seven_component.py" --episode 14
```

Use `--checkpoint`, `--training-report`, `--evaluation` and `--episode-dir` to
stage a later successful refinement explicitly. The wrapper checks the original
ZIP hash, unchanged object targets and the HaWoR hand report. It replays the
selected actor, calls the existing component/portable builders, validates every
physical state and action, and runs the unchanged common evaluator. Outputs are
under `components/`, `common_components/` and `portable_archives/` in this new
experiment only. Optional `--video` adds a reviewed video to the component.

The existing stage helper restricts input runs to a `refinement_20261018`
subdirectory. The wrapper changes only its in-memory base-directory variable
and stages its fresh replay in this experiment's own directory of that name;
the old stage helper, actual runtime code and old outputs remain unchanged.

Run with the DexAgent packaging environment, whose requirements include the
official kit dependencies, including `kaggle>=2` even for offline preflight:

```bash
TASK_PY=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python
TASK_BASE=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019
TASK_OUTPUT=/data1/SRL_V2D/challenge_submissions/track3_full_20261019/all_20_episodes
TASK_ARGS=()
for TASK_EPISODE in 14 15 16 20 25 28 30; do
    printf -v TASK_NAME 'episode_%06d' "$TASK_EPISODE"
    TASK_ARGS+=(--component "$TASK_EPISODE=$TASK_BASE/common_components/$TASK_NAME")
    TASK_ARGS+=(--portable "$TASK_EPISODE=$TASK_BASE/portable_archives/${TASK_NAME}_portable.tar.gz")
done
"$TASK_PY" "$TASK_BASE/package_code/assemble_twenty.py" \
    "${TASK_ARGS[@]}" --output "$TASK_OUTPUT" --archive
```

Archive paths are examples: pass the actual completed archive paths. Optional
`--video EPISODE=PATH` adds a reviewed video or directory. Otherwise, the assembler
copies a component's `videos/` when available. Videos are supporting evidence,
not replacements for recorded physical states and robot actions.

The script uses the unchanged official packer to produce one combined
`track3_submission.parquet`, containing all 9,120 pose rows and all object meshes.
It then runs `tools/preflight.py --skip-kaggle` and the official uploader's local
`validate_submission` function without calling its authentication or upload code.
It rejects changed old artifacts, missing episodes, inconsistent producing code,
incorrect checkpoint attribution, invalid meshes and incomplete frame grids.

To recheck the existing thirteen and official kit without making a release:

```bash
"$TASK_PY" "$TASK_BASE/package_code/assemble_twenty.py" --audit-existing-only
```

`--preparation-commit-url` records a later packaging commit separately. The
producing evaluator commit remains the one that actually ran each controller.
The final manifest explicitly distinguishes local payload validation from
Kaggle account readiness, organizer runtime certification and hidden-GT scores.
