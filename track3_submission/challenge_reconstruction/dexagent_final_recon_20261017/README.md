# DexAgent trial on reconstructed episodes with floating Sharpa hands

This experiment tests episodes **1, 3, 5, 8, and 10** from
`/data1/SRL_V2D/challenge_reconstruction/final_recon`. It executes the existing robot
reference in physics, runs DexAgent's released predictive-sampling MPC refinement, and
executes the retained reference again from the identical initial state.

The upstream MPC module is unchanged at DexAgent commit
[`e29bdad8ed3e95b2c9ee2eea9044a6b4115ec633`](https://github.com/dexagent1/DexAgent/commit/e29bdad8ed3e95b2c9ee2eea9044a6b4115ec633).
`floating_runtime.py` and `floating_tracking.py` port its runtime and reference interface
to the requested floating hands. This is **a test of the released MPC module with a
floating-Sharpa port**, not the complete native LLM-driven DexAgent pipeline or its YAM-arm
certification. No LLM API key or GPU is needed for this CPU MuJoCo experiment.

## Grasp refinement follow-up (2026-10-17)

The current priority is stable pickup and holding across the same **five episodes**.
The initial pilot is **0/5 reference-proxy MP-SR**; the completed CEM and block-polish
follow-ups are **1/5**, with episode 8 passing. Original `final_recon` inputs are preserved.
See the [five-episode review](refinement_20261017/metrics_review.md) for per-object errors,
grasp retention, complete comparison videos, and independent replay checks. Passing the
mean-error metric does not establish a slip-free grasp or reproduce every peak motion.

The follow-up separates failures to reach the object, failures to form opposing
finger contact, and slipping after pickup. See
[`refinement_20261017/GRASP_DIAGNOSIS.md`](refinement_20261017/GRASP_DIAGNOSIS.md) and
[`refinement_20261017/grasp_stability/README.md`](refinement_20261017/grasp_stability/README.md).
Controller-only tuning improved wrist tracking but did not solve the grasps. Geometry
seeds are proposals until their complete physics executions pass; fingertip proximity
alone is not evidence of a stable grasp.

`code/wrist_reference.py` repairs equivalent Euler-angle branches without changing
physical wrist poses. `code/refine_grasp_cem.py` searches 56 independent wrist/finger
corrections with bounded servos and unactuated objects. This follow-up optimizer is
custom code, separate from the unchanged upstream MPC pilot. It preserves object
targets, original timing, and exact scene initial state. It uses CPU processes.
Search and replay both load the frozen XML and reset its exact saved keyframe.
Superseded `cem_smoke` and interrupted `cem_v1` used a rounded reference initial state;
their search improvements are not reliable final-execution evidence.

The corrected full search is `cem_v2` (Slurm array **47552**), with independent
audit/render array **47556**. Each episode's `result.json`, `mp_sr.json`, and
`audit/search_replay_consistency.json` distinguish task success from replay consistency.
These trajectory-optimization trials are not learned-policy checkpoints. The separate
[submission audit](refinement_20261017/submission_readiness/AUDIT.md) records what remains
before preparing a genuine five-episode policy-evaluation package.

The later `block_polish` runs preserve objects that already satisfy the pose thresholds
while testing smaller wrist/finger corrections. `feedback_polish` tests wrist corrections
inferred from measured object error only during observed thumb-plus-finger contact. Every
proposal is executed again in coupled two-hand/two-object physics; improving one object
does not justify dropping the other. Source object trajectories are never edited to
increase the evaluation score.

The lid and spoon diagnostics found that initial contact can push or rotate an object
away before another finger reaches it. Two fingers touching is also insufficient: in a
lid trial their actual contact normals pointed toward the same side. The isolated
`contact_acquisition_runtime.py` and `contact_retention_runtime.py` experiments therefore
record actual contact forces/geometry and test robot feedback during pickup and transport.
They command only the 56 existing robot actuators and record every 500 Hz control for
independent replay. They are experimental controllers, not successful learned policies.

To refresh the completed-run review or inspect one full-search audit:

```bash
.venv/bin/python code/summarize_refinement_audits.py
.venv/bin/python code/check_cem_run.py --help
```

Further experiments use separate output directories (`grasp_transfer`, `feedback_polish`,
`block_continue_s*`, and acquisition/retention trial directories); failed proposals are
retained for diagnosis. Do not promote these folders merely because their replay passes.

## Embodiment and metric

- Two Sharpa Wave hands use the reconstruction pipeline's exact hand geometry, collision
  meshes, and inertias. Each has 6 wrist and 22 finger controls: **56 robot controls** total.
  Position servos have bounded targets and forces. Objects have unactuated free joints;
  their reference poses are targets for evaluation, never commands that move them.
- The upstream search uses **14 residual coordinates per spline knot**: 6 wrist coordinates
  and one shared finger-flexion offset per hand. It does not independently optimize all
  44 finger joints. Default search: 16 candidates, 0.4-second horizon, replanning every
  0.2 seconds, 4 CPU workers, seed 44, and a 10-minute search budget per episode.
- Each episode retains its original frame count and timing. Search may exhaust its budget
  before reaching the end; the exported reference and final physics evaluation still cover
  the full episode. The 10-minute budget excludes some preparation and final execution work.
- A finite circular tabletop is fitted from the calibrated first image. The fit assumes a
  horizontal round wooden table; height uses the initial object-mesh lower bound and
  thickness is assumed to be 4 cm. Review `stage2/table_fit.jpg` and `input_manifest.json`.
  Friction is assumed; object masses and inertias come from the source assets.
- **Reference-proxy MP-SR** is the primary success metric: every object must have mean centre
  position error **<3 cm** and symmetry-aware mean orientation error **<30 degrees** over the
  entire original episode. Reconstructed reference poses substitute for confidential GT;
  this is **not an official leaderboard score**. Physics verification is reported separately.

[METRIC.md](METRIC.md) documents official symmetry classes, estimated local mesh axes,
full-horizon validation, and the distinction between `reference_proxy_mp_sr` (one official
initial-scene alignment) and `declared_frame_mp_sr` (only the declared adapter transform).

## Prepare or reproduce on another server

The current scripts assume the repository is located at **`/data1/SRL_V2D`** and this
workspace remains under `challenge_reconstruction/dexagent_final_recon_20261017`.
Keep that layout, or explicitly update the root in `code/prepare_episodes.py`, Slurm paths,
and source `config.json` video paths. Regenerate scenes on the destination server because
generated XML and manifests contain absolute asset paths.

Copy this workspace's `code/`, `requirements.lock.txt`, and `metric_sources/`, together with:

1. The five complete `final_recon/episode_XXXXXX` bundles, including meshes, collision parts,
   URDFs, `reference.parquet`, `object_poses.npz`, camera data, calibration, and configuration.
2. The source videos addressed by each bundle's
   `config.json` → `source` → `anycalib/undistorted.mp4`. These may lie outside `final_recon`.
3. `robotic_grounding/source/robotic_grounding/robotic_grounding/assets`, including Sharpa
   XML, URDF, and all referenced mesh files.
4. DexAgent at the pinned commit under `third_party/DexAgent`. For a fresh destination:

```bash
cd /data1/SRL_V2D
git clone https://github.com/dexagent1/DexAgent.git third_party/DexAgent
git -C third_party/DexAgent checkout e29bdad8ed3e95b2c9ee2eea9044a6b4115ec633
```

The tested environment uses Python 3.11. MuJoCo version and dependency versions are pinned in the
lock file. Rendering also needs `ffmpeg` and a working software EGL/Mesa or OSMesa installation;
physics and metric computation do not require a graphics context.

```bash
cd /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
export PYTHONPATH=/data1/SRL_V2D/third_party/DexAgent:$PWD/code
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

.venv/bin/python code/prepare_episodes.py --episodes 1 3 5 8 10
for episode in 000001 000003 000005 000008 000010; do
  .venv/bin/python code/evaluate_mp_sr.py \
    --reference-dir /data1/SRL_V2D/challenge_reconstruction/final_recon/episode_$episode \
    --prepare-geometry --output episodes/episode_$episode/metric_geometry.json
done
```

Preparation rebuilds input scenes: run it before launching experiments. Preserve completed
run directories and their frozen scenes. The metric evaluator rejects changes to source
object poses or meshes after their hashes have been frozen.

## Verify and run

```bash
.venv/bin/python -m pytest -q code/tests/test_evaluate_mp_sr.py
.venv/bin/python code/tests/test_floating_runtime.py episodes/episode_000001/stage2/scene.xml

# One complete experiment, using an unused tag:
.venv/bin/python -u code/run_episode_test.py \
  --episode 1 --seed 44 --workers 4 --samples 16 --max-minutes 10 --tag manual_v1

# Five-episode Slurm array, up to five concurrent CPU jobs:
mkdir -p logs
DEX_TEST_TAG=pilot_v2 sbatch --export=ALL,DEX_TEST_TAG=pilot_v2 code/test_episodes.sbatch
```

Adjust partition, CPU, memory, and log paths in the batch file for another cluster. The
provided batch script requests 4 CPUs and 16 GB per task; it requests no GPU. Use a fresh
tag for another trial: existing trajectory recordings are deliberately not overwritten.
The earlier `pilot` attempt failed preflight; its outputs are retained for diagnosis.
Current reruns use `pilot_v2`.

## Read results and make videos

```bash
.venv/bin/python code/summarize_results.py --tag pilot_v2 --seed 44
```

This writes `reports/pilot_v2_seed44.json` and `.md`: per-object baseline/refined errors,
MP-SR, independent physics replay, search cost, elapsed time, and the actual retained wrist
and finger corrections. Missing/error runs remain visible. The full five-episode score
stays pending until every requested episode has complete metrics. Reference self-checks
under `metric_sources/` validate metric plumbing and are never counted as task results.

Each experiment is stored in `runs/<tag>/episode_XXXXXX/seed_44/`:

| File or directory | Contents |
|---|---|
| `status.json` | Current stage or error traceback |
| `result.json` | Completed baseline/refined experiment and provenance |
| `baseline/`, `refined/` | Measured `trajectory.npz`, frozen `scene_physics.xml`, `execution_report.json`, and `mp_sr.json` |
| `mpc/report.json` | Search settings, candidate costs, budget stop, and retained/rejected refinement |
| `mpc/reference.npz` | Retained robot commands; baseline commands if refinement was rejected |

`cost_after` in the upstream search report describes the proposed candidate, including a
rejected candidate. The summary distinguishes proposed and selected costs. Search cost is
not MP-SR. It also separates wrist translations in centimetres from finger angles in degrees;
the upstream `residual_max.arm_rad` field mixes units in this floating-hand port.

To render calibrated original/reference/baseline/refined panels after completion, create
`render_args.json` with paths to the episode and its two execution reports:

```json
{
  "episode_dir": "episodes/episode_000001",
  "baseline_report": "runs/pilot_v2/episode_000001/seed_44/baseline/execution_report.json",
  "refined_report": "runs/pilot_v2/episode_000001/seed_44/refined/execution_report.json",
  "out_dir": "runs/pilot_v2/episode_000001/seed_44/videos",
  "stride": 2
}
```

Run `.venv/bin/python code/render_rollouts.py --args render_args.json` with software EGL
environment variables matching the batch script, or `MUJOCO_GL=osmesa` on a host with
OSMesa installed. Output is `videos/reference_baseline_refined.mp4`, still images, and
`render_report.json`. The kinematic reference is labelled separately from measured physics;
all panels retain the source timestamps and moving camera.

## First completed test: seed 44, `pilot_v2`

Slurm array **47407** completed all five episodes. Baseline and refined execution both
scored **0/5 reference-proxy MP-SR**. All original timelines were evaluated; all MPC
planning steps completed without exhausting the ten-minute budget. Each episode took
1.54–3.79 minutes including baseline, search, refined execution, scoring and replay checks.
See [the full per-object report](reports/pilot_v2_seed44.md) and
[machine-readable results](reports/pilot_v2_seed44.json).

The retained references changed wrist commands by 0.68–1.78 cm on average across hands
and episodes. Every search reduced its native weighted objective, but that objective
includes force/penetration penalties and is not MP-SR. A retained change therefore does
not imply improved task success. This is one seed of the adapted MPC module, not a
complete native DexAgent pipeline comparison.

Two scene/input limitations matter when reading these results. Episode 1 starts with
about 23 mm of pot/lid collision overlap already present in the supplied meshes/poses;
we preserved that source data. Episode 10 starts with both cups correctly supported,
then knocks them off the table around five seconds. The scene has no room floor, so
continued falling inflates its multi-metre mean position errors. Episode 8 also drops
the cup late in the rollout. See the [support/drop audit](runs/pilot_v2/support_failure_audit.json).
Independent control replay verifies that recorded motion was actually simulated; it
does not certify successful manipulation or penetration-free motion.

Comparison videos show original RGB, the kinematic reference, baseline physics and
refined physics:

- [Episode 1](runs/pilot_v2/episode_000001/seed_44/videos/reference_baseline_refined.mp4)
- [Episode 3](runs/pilot_v2/episode_000003/seed_44/videos/reference_baseline_refined.mp4)
- [Episode 5](runs/pilot_v2/episode_000005/seed_44/videos/reference_baseline_refined.mp4)
- [Episode 8](runs/pilot_v2/episode_000008/seed_44/videos/reference_baseline_refined.mp4)
- [Episode 10](runs/pilot_v2/episode_000010/seed_44/videos/reference_baseline_refined.mp4)
