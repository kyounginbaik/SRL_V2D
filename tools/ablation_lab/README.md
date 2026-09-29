# Track 3 Ablation Lab

A local, read-only A/B workbench for the downloaded V2D Track 3 dataset. It uses
actual saved model outputs, regenerates MANO geometry for every frame on CPU,
and measures public object trajectories with explicit coordinate alignment.
It does not train RL policies.

## Open

The server started during this audit is at **http://node100:8765**. On the host,
use **http://localhost:8765**. If the cluster hostname is not reachable from your
computer, forward the port through your usual SSH connection:

```bash
ssh -L 8765:node100:8765 <your-cluster-login>
# Then open http://localhost:8765 on your computer.
```

Start or restart from the repository root:

```bash
/home/exaFLOPs/miniforge3/envs/v2d/bin/python -m tools.ablation_lab.serve \
  --host 0.0.0.0 --port 8765
```

The default bind address is loopback if `--host` is omitted. The server has no
job-launch endpoint. It serves generated artifacts and the explicitly selected
dataset's `runs/` and `track_3/public/` directories, with HTTP byte ranges for video.

## What to compare

Select an episode, choose A or B in the left sidebar, then select a pipeline
configuration. Presets isolate useful comparisons:

- **Gsplat on/off:** saved joint refinement versus its unrefined scene.
- **Before/after depth fit:** calibrated HaMeR before depth shift versus fitted observations.
- **Observed/gap-filled:** actual observation records versus filled tracks.
- **Before/after cleanup:** the user's saved outlier-cleaned tracks, where present.
- **Reconstruction/loader scale:** isolates the per-hand scale the current loader discards. This is a diagnostic geometry view, not a full loader rerun.
- **Static/moving camera:** saved packaging variants; no claim of independent model inference.

The **Lens**, **Depth**, and **Masks** tabs expose the saved upstream outputs.
Depth previews use a fixed 0.1–2.0 metre colour scale across frames (values outside
that range are clipped for display only).

The timeline synchronizes frame indices across source video, 3D geometry and
recorded overlays using each video's encoded FPS. Space plays/pauses; arrows
step frames. Click the chart or **Jump to largest outlier**. The fit button in a
3D pane includes all current-frame geometry, including far-away outliers.
The default camera frames typical motion without removing outliers from the data.

**Kinematic replay is independent of reconstruction selectors.** Its input is
identified by numerically comparing stored MANO targets against available
reconstruction bundles. Episode 41 predates cleanup; episode 31 matches the
cleaned input. Do not infer that switching HaMeR to another model recomputes IK.

Missing combinations show **Not computed**, with no borrowed metrics or video.
Registered HaWoR runs are selectable when present. SAM3D, factory lens
calibration and stereo depth still require separate runs or input adapters.

## Build and audit

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/exaFLOPs/miniforge3/envs/v2d/bin/python -m tools.ablation_lab.build
```

Inputs default to `/data1/video_to_data_challenge` and are read-only. Generated
artifacts live under `.ablation_lab/site/` (about 0.65 GB for these 20 episodes, including depth previews).
The builder needs NumPy, SciPy, OpenCV, trimesh, PyArrow and Pillow, already in
the existing `v2d` environment. MANO assets and vendored Three.js are reused
from the existing local runs. No model weights are downloaded by the builder.

Outputs include `REPORT.md`, `audit.csv`, `audit.json`, `catalog.json`, and one
`episode.json` with per-frame binary meshes for each episode. Archived results
are explicitly marked, and incomplete episodes are never called complete.

**Metrics are not official challenge scores.** Object path RMSE uses one rigid
all-frame fit of predicted centres to visible public object poses, with no scale
fit. This is an optimistic diagnostic for trajectory shape. The report also
contains first-visible-pose-anchored errors in each episode JSON. Missing poses
are masked, not interpreted as zero rotations. There is no hand ground truth;
the 1.1 m wrist-distance threshold is only an inspection heuristic.

## Add real model ablations

Plan isolated, one-factor experiments (does not execute inference):

```bash
PY=/home/exaFLOPs/miniforge3/envs/v2d/bin/python
$PY -m tools.ablation_lab.experiments plan --episode 41 --factor hand
$PY -m tools.ablation_lab.experiments plan --episode 41 --factor refine
$PY -m tools.ablation_lab.experiments plan --episode 41 --factor mesh
$PY -m tools.ablation_lab.experiments status
```

Supported factors are `hand`, `refine`, `mesh`, `lens`, `gravity`, and `camera`.
Each configuration receives a distinct output directory and exact argument
list in `.ablation_lab/experiments.json`. Reference frame 0 is held fixed in
these planned comparisons. Camera comparisons hold gravity off in both runs.

On a node with the needed Docker images, weights and an allocated GPU:

```bash
V2D_DOCKER_GPUS=<allocated-device> $PY -m tools.ablation_lab.experiments run \
  --id <id-printed-by-plan>
```

Execution records the command, Git commit, status and logs, validates the actual
hand/refinement provenance, and registers the completed scene in the website.
Rebuilding the base site preserves completed registered experiments. A run
completed outside this command can be registered with `register --id <id>` after
placing its output in that experiment's directory. Refresh the page afterward.
Generated SAM3D meshes do not receive scan-frame object metrics without an
explicit frame mapping; those numbers remain unavailable.

HaWoR requires its `v2d_hawor` image and weights. Its adapter maps HaWoR's
internal 30 fps timeline back to the source frame count before depth alignment.
Factory calibration and stereo depth are deliberately marked as adapter work,
not advertised as CLI switches. Experiments are one-factor comparisons rather
than a Cartesian sweep.

### Add another ablation

Keep each new option end-to-end and provenance checked:

1. Add its value to `FACTORS` (and the default to `BASE` when appropriate) in
   `experiments.py`.
2. Extend `make_command()` with the real pipeline CLI argument. Never represent
   an unavailable model by relabeling another model's output.
3. Add or expose the corresponding pipeline argument in
   `run_ego_reconstruction.py` / `run_ego_wilor.py`. Write the selected model
   and relevant settings into `manifest.json` under `sources`.
4. Extend `validate_result()` so registration rejects a bundle whose recorded
   provenance does not match the requested configuration.
5. Add the option to `definitions` in `web/app.js`. If one selection constrains
   other controls, load the completed experiment configuration as the HaWoR
   selector does.
6. Add a focused unit test in `test_lab.py` and a real browser interaction in
   `browser_test.py`, then run the verification commands below.

Typical workflow:

```bash
$PY -m tools.ablation_lab.experiments plan --episode 41 --factor hand
$PY -m tools.ablation_lab.experiments run --id <printed-id>
# Or, after an external run populated the planned output directory:
$PY -m tools.ablation_lab.experiments register --id <printed-id>
```

Runs, weights, datasets, browser downloads and generated website artifacts live
under ignored paths and must not be committed. Share the command and manifest;
regenerate the site locally from the same dataset checkout.

## Retargeting reproducibility test

A separate **fresh HaMeR crop test** on episode 41 frame 244 reproduced the
12.53384 m outlier with the baseline SAM2 crop, and produced 0.61784 m with the
WiLoR detector crop. The model weights and 1.7 crop expansion are identical;
only the crop box changes. The site exposes the evidence with **View fresh crop
ablation** in episode 41. This is one frame, before depth fitting, and is not a
full-sequence improvement claim. Reproduce with `crop_probe.sh` on two scheduled
CPU cores, then `python -m tools.ablation_lab.export_probe` to publish the evidence.

The original baseline's episode 41 IK was freshly rerun on CPU using the saved
loaded MANO input. It exactly matched saved wrist positions, finger joints and
IK errors. Results: `.ablation_lab/site/retarget_retest.json`; log:
`.ablation_lab/retarget_retest_native.log`. This does not constitute a fresh
neural reconstruction run.

```bash
srun --nodelist=node01 --gres=gpu:0 --cpus-per-task=2 --time=00:20:00 \
  bash tools/ablation_lab/retarget_smoke.sh episode_000041
```

The test uses the image's native Python launcher and Isaac group membership,
mounts original inputs read-only, and writes only to `.ablation_lab/retarget_retest`.

## Verification

```bash
$PY -m unittest tools.ablation_lab.test_lab -v
.ablation_lab/venv/bin/python -m tools.ablation_lab.browser_test \
  --browser .ablation_lab/browser/chrome-headless-shell-linux64/chrome-headless-shell
```

The browser tests exercise WebGL, numerical displays, outlier navigation,
asynchronous episode changes, unavailable-model handling, actual video seeks,
IK provenance, mobile overflow and range requests. Screenshots and results are
under `.ablation_lab/browser_test/`.
