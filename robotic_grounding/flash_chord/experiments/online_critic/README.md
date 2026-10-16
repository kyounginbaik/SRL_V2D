# UniRetarget online critic and stable-control training

This recipe supports one episode or an explicit batch with two floating Sharpa hands.
The current comparison uses episode 1 from `challenge_reconstruction/final_recon`,
with one seed per upper-refinement variant. Other episodes are not scheduled.
Preparation snapshots source, launchers, reference data and the exploration prior
into a fresh directory. Each launch checks SHA-256 manifests before starting.

## Current episode-1 comparison

| Variant | Upper refinement | Lower control and exploration |
| --- | --- | --- |
| `full_edit` | 56 wrist/finger coefficients | All 56 actions; EigenDEXplore finger noise |
| `wrist_only` | 12 wrist coefficients: 6D pose per hand | All 56 actions; EigenDEXplore finger noise |

Both use seed 43, identical `final_recon/episode_000001` targets, the same measured
support surface, controller limits, replay and training budget. The wrist-only upper
network truly outputs 12 coefficients; it cannot edit finger references. Lower finger
commands remain full dimensional, including independent correction outside the PCA span.

Episode 1 includes a tabletop and floor fitted from its calibrated stereo depth.
The scene uses those measured planes without shifting hand or object targets. The table
has a fixed root: representing it as a free body with very large artificial inertia caused
unstable initial contacts. The fixed table keeps the pot within 1.1 cm of its initial
position in a two-second CPU check, but the initial scene is still unsuitable for full
training: a no-hand physics rollout reproduces sustained lid rotation above the 20-degree
tracking threshold. Progress reports place that failure at 0.10 s; accounting for the
post-step measurement gives 0.15 s of physical simulation. Existing pot/lid collision
proxy overlap and initialization require validation before full training. The measured
support fit alone does not establish a stable initial object configuration. Fit provenance,
diagnostic scripts and trajectories are copied into
`frozen/episode_000001/assets/support_validation/`.

Both comparison jobs completed their 1,048,576-transition smoke training and then failed
while evaluating the archived best policy because its robot-reference snapshot omitted
the object/contact command digest. The archive writer now stores that digest. Legacy
archives load only when their original `pair.json` matches the snapshot checksum and
identity and its command digest matches the actual reference. Evaluation/export can be
retried from those immutable policies without repeating smoke training. This metadata
fix does not fix the physical initialization failure, and full training remains pending
collision/initialization validation.

The existing exploration implementation follows [EigenDEXplore](https://eigendexplore.github.io/)
with the published Sharpa basis (nine directions per hand), independent joint noise,
and learned eigen-noise scales. This repository adapts the covariance to a tanh-Gaussian
SAC residual policy, with covariance-aware action density and entropy. It is not a
reproduction of the paper's PPO benchmark. Initial finger variance is split equally
between independent and eigen noise; wrists receive no eigen perturbation. Basis hashes,
joint ordering and source revision are stored with the prepared prior.

The current portable input bundle on the original server is
`/data1/SRL_V2D/challenge_reconstruction/uniretarget_ep01_stable_20261016/episode1_final_recon_inputs.tar.gz`
(6 MB), with an adjacent `.sha256` file. It contains the exact reference and asset
closure, prior, measured support evidence and public participant packer subset.
Archive SHA-256: `4dca9d75288bddaee6f22824a0486493cc1d57206af7b2801eeb749cee0c53b2`.
Copy both files, verify with `sha256sum -c episode1_final_recon_inputs.tar.gz.sha256`,
then extract with `tar -xzf episode1_final_recon_inputs.tar.gz`.
After extraction, resolve the inventory paths for that server without changing the inputs:

```bash
python3 - /absolute/path/to/episode1_inputs <<'PY'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
for name, checksum in json.loads((root/'sha256.json').read_text()).items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest() == checksum, name
inventory = json.loads((root/'episode1-inventory.json').read_text())
for item in inventory['episodes']:
    for key in ('source_reference', 'source_bundle'):
        item[key] = str(root/item[key])
(root/'local-inventory.json').write_text(json.dumps(inventory, indent=2)+'\n')
PY
```

Use `episode1_inputs/local-inventory.json`, `episode1_inputs/priors`,
`episode1_inputs/submission_kit`, and `episode1_inputs/measured-support` below.

After collision/initialization validation passes, prepare two fresh directories from an
inventory containing only episode 1:

```bash
for scope in full wrists_only; do
  python3 robotic_grounding/flash_chord/experiments/online_critic/prepare_batch.py \
    --inventory /absolute/path/to/episode1-inventory.json \
    --priors-root /absolute/path/to/priors --kit /absolute/path/to/v2d_submission_kit \
    --support-root /absolute/path/to/measured-support \
    --upper-edit-scope "$scope" --seed 43 --run-root "/absolute/path/to/$scope-run"
  bash "/absolute/path/to/$scope-run/run.sh" preflight
  sbatch --output="/absolute/path/to/$scope-run/slurm/%j.out" \
    --error="/absolute/path/to/$scope-run/slurm/%j.err" \
    "/absolute/path/to/$scope-run/cluster/train_full.sbatch" "/absolute/path/to/$scope-run" 43
done
```

This submits exactly two GPU jobs. Each validates integration, starts its separate full
training phase, exports actual unassisted episode-1 trajectories and writes readiness
reports. The other 19 public episodes remain explicitly missing from a combined submission.
The general batch utility below remains available for later expansion.

## Clone and build

Use Linux, host Python 3.11+, Docker with NVIDIA Container Toolkit, and Git LFS.
The pinned image uses CUDA 13 JAX wheels and requires NVIDIA driver >=580.
The production allocation is one 48 GiB-class GPU, 12 CPUs and 96 GiB host RAM;
the container limit is 90 GiB. Allow substantial storage for checkpoints and videos.

```bash
git clone --branch uniretarget/online-critic-full-training \
  git@github.com:kyounginbaik/SRL_V2D.git
cd SRL_V2D
git lfs install
git lfs pull
python3 -m venv .venv-uniretarget-prepare
source .venv-uniretarget-prepare/bin/activate
python -m pip install numpy==2.2.5 pyarrow==22.0.0
cd robotic_grounding/flash_chord
docker build -f docker/Dockerfile --target dev -t flash_chord:uniretarget-base .
docker build -f experiments/online_critic/Dockerfile \
  -t flash_chord:uniretarget-online .
cd ../..
```

The second image adds pinned OpenCV for checkpoint overlays and pandas for the official
submission packer without replacing numpy. On the original server, set
`UNIRETARGET_IMAGE=flash_chord:uniretarget-ep01-stable-20261016` to use the tested
image. Runtime code comes from the prepared snapshot.

## Legacy single-episode input bundle

The bundle in this section belongs to the earlier prepared-reference experiment.
For the current comparison, use the exact `final_recon` episode-1 inventory, assets,
measured support and prior described above; do not substitute this older bundle.

Inputs are separate from Git. Copy these files from the original server:

```text
/data1/SRL_V2D/challenge_reconstruction/uniretarget_portable_20261016/episode_000001_inputs.tar.gz
/data1/SRL_V2D/challenge_reconstruction/uniretarget_portable_20261016/episode_000001_inputs.tar.gz.sha256
```

The bundle is approximately 4 MiB; SHA-256:
`cfe540b7d2fcde752c8689ee1fc52f66a3dc9ddb597de0027f84ab071973f093`.
It includes the 335-frame, 20 Hz reference, pot/lid meshes and collision assets,
source reference, preparation metadata and Sharpa eigen prior. Pot/lid masses
are 0.2/0.1 kg. Preserve all supplied paths and bytes.

On the destination, replace `SOURCE_SERVER` with your SSH host:

```bash
mkdir -p /absolute/path/to/episode-inputs
cd /absolute/path/to/episode-inputs
scp SOURCE_SERVER:/data1/SRL_V2D/challenge_reconstruction/uniretarget_portable_20261016/episode_000001_inputs.tar.gz .
scp SOURCE_SERVER:/data1/SRL_V2D/challenge_reconstruction/uniretarget_portable_20261016/episode_000001_inputs.tar.gz.sha256 .
sha256sum -c episode_000001_inputs.tar.gz.sha256
tar -xzf episode_000001_inputs.tar.gz
```

Preparation independently checks every file against the committed `data_manifest.json`
and rejects unresolved Git LFS assets. The loader and video metrics resolve obsolete
producer asset paths against the bundled sibling `assets/` directory. Reference
bytes and physical targets stay identical. No original-server mount is needed.

## Prepare and check

From the repository root, choose a nonexistent run directory outside the FlashCHORD
source tree:

```bash
export RUN_ROOT=/absolute/path/to/uniretarget-run
python3 robotic_grounding/flash_chord/experiments/online_critic/prepare.py \
  --data-root /absolute/path/to/episode-inputs --run-root "$RUN_ROOT"
bash "$RUN_ROOT/run.sh" preflight
```

CPU preflight composes the actual Hydra commands for smoke and full training, all
three seeds, identical physics, 2,048 worlds, bounded adaptation, and the new critic
schedule, and assembles a complete one-world CPU scene to check asset loading.
It saves `reports/full_launch_preflight.json` and `launch_resolved.yaml`
in the per-seed run directories. GPU integration remains a separate check.

## Launch

Without Slurm, run one seed per available GPU in persistent terminal sessions:

```bash
GPU_DEVICE=0 bash "$RUN_ROOT/run.sh" full 43
# In separate sessions on two other available GPUs:
GPU_DEVICE=1 bash "$RUN_ROOT/run.sh" full 44
GPU_DEVICE=2 bash "$RUN_ROOT/run.sh" full 45
```

`GPU_DEVICE` accepts a Docker GPU index or UUID. Under Slurm the launcher maps
`SLURM_JOB_GPUS` to a UUID. Numeric allocations are interpreted as NVIDIA device
minor numbers, matching the original cluster; adapt `cluster/allocated_gpu.py`
if your cluster uses a different Slurm GRES convention.

For Slurm, select your cluster's partition and permitted time limit:

```bash
for seed in 43 44 45; do
  sbatch --partition=YOUR_GPU_PARTITION --time=7-00:00:00 \
    --output="$RUN_ROOT/slurm/seed_${seed}_%j.out" \
    --error="$RUN_ROOT/slurm/seed_${seed}_%j.err" \
    "$RUN_ROOT/cluster/train_full.sbatch" "$RUN_ROOT" "$seed"
done
```

The script requests 1 GPU, 12 CPUs and 96 GiB RAM. Export `DOCKER_USE_SG=1`
before submitting if your Slurm session needs `sg docker` for socket access.
The wrapper stops its container on termination.

Each `full` job performs CPU unit checks, a fresh **1,048,576-main-transition** GPU
integration run with two upper rounds, then verifies real intermediate critic fitting,
finite parameters, matched candidate/control training, measured feedback and media.
If those pass, it automatically starts a separate fresh **100,663,296-main-transition**
run. For compatibility, this full output phase is named `pilot`.

Rejected edits are normal outcomes; acceptance is not required to pass integration.
`smoke` runs only integration. `pilot` starts the full budget directly for standalone runs; prepared batches require
a passed integration report for that episode and seed. Use `full` for standalone runs on a new server. Do not restart a trained seed/phase:
complete joint upper/lower resume is unsupported. Adaptation and evaluation add
simulation work beyond the main-transition budget.

W&B is disabled by default. To enable it with your account:

```bash
export WANDB_MODE=online WANDB_ENTITY=YOUR_ENTITY WANDB_PROJECT=YOUR_PROJECT
# Authenticate through host ~/.netrc, WANDB_NETRC or exported WANDB_API_KEY.
export UNIRETARGET_RUN_TAG=my-unique-run-name
GPU_DEVICE=0 bash "$RUN_ROOT/run.sh" full 43
```

Credentials remain outside Git and the input bundle. Use unique tags for distinct
experiments; the logger appends seed and phase to each run identifier.

## Method and outputs

The upper actor receives reference/correction summaries, signed hand-tracking errors,
execution quality, window/failure timing and local per-object pose errors. Its output
is 56 local correction coefficients for `full` or 12 wrist coefficients for `wrists_only`.
The cost critic receives lower observations,
critic context and lower robot actions. Four intermediate TD updates run every
16 vector steps after the first upper round, requiring at least 256 valid recent
rows and filtering by current reference and outer generation. Existing outer-round
and measured unassisted-return fitting remain enabled.

Main training pauses every 501,760 transitions for up to four proposals. Proposals
use three differentiable lower-actor Adam steps and four upper updates. Physical
candidate/control adaptation uses three matched checkpoints, capped at 448 vector
steps per branch (917,504 transitions). Physical acceptance and frozen confirmation
remain mandatory. See [method details](../../docs/wrench_hypergradient.txt) and
[critic diagnostics](../../docs/critic_diagnostics_20261016.txt).

Under `episode_000001/runs/{smoke,pilot}/episode_000001/chunk/seed_N/`:

| Artifact | Meaning |
| --- | --- |
| `run_config.json`, `stage.json` | Command, metadata and phase status. |
| `training/online_critic_updates.jsonl` | Actual intermediate optimizer counts, generation, reference and losses. |
| `training/upper_updates.jsonl` | Proposals, adaptation checkpoints, `critic_pair` diagnostics and rejection/acceptance decisions. |
| `training/search_batches/step_*/` | Upper states, trial trajectories and pre-proposal critic snapshots. |
| `training/references/`, `training/policy_*.safetensors` | References and corresponding lower policies. |
| `checkpoint_videos/`, `voc_checkpoint_videos/`, `reference_overlays/` | Unassisted/current-assistance videos and reference overlays. |
| `evaluation/metrics.json` | Unassisted evaluation of the archived best policy/reference pair. |

The run root saves `reports/smoke_validation_episode_000001_sN.json` with actual
integration results. Logs go to the terminal or Slurm paths. Share reference snapshots
and metadata with policy checkpoints.

More critic fitting is not yet proven to improve task success. Paired diagnostics
compare predictions/returns on realized trajectories, not the fixed-state hypergradient.
Fixed-start evaluation does not establish generalization to other resets or episodes.

## Controller changes and limits

The recipe uses wrist residual bounds of ±5 cm per position coordinate and ±0.5 rad
per RPY coordinate. Post-mapping, post-EMA correction changes are limited separately
per hand to 0.15 m/s and 1.5 rad/s in RPY coordinates. The reference is added afterward:
intentional motion, including episode 1's shaking at 6–10 seconds, is not smoothed by
this limiter. Finger correction range stays unchanged. These settings are opt-in;
the base action configuration keeps the limiter disabled.

Batch preparation sets physical wrist speed caps to at least 1 m/s and 12 rad/s,
using 1.2 times each episode's peak reference speed, up to ceilings of 2 m/s and
20 rad/s. The ceilings prevent reconstruction jumps from permitting extreme speeds.
Reference peaks, resolved caps and any uncovered reference transitions are recorded
in each frozen manifest. Episode 1's 6–10 second shaking segment is fully covered;
its numerical targets remain unchanged. Its late right-hand reconstruction gap has
two linear and one angular transitions above the physical ceilings. The lower actor observes
object body velocity, allowing it to detect motion associated with slip. This is a
fresh-policy input change; do not load an older policy with the new observation shape.

A historical lower-policy run grew more aggressive while assistance remained at 1
and no upper edits were accepted. This supports testing tighter residual control;
it does not prove these settings improve grasp success. GPU integration and actual
unassisted rollouts remain required. No automatic assistance decay is forced.

## One seed across the selected episodes

`prepare_batch.py` consumes an inventory with source reference and bundle paths,
the Sharpa prior directory, and the official participant submission kit. Inputs are
external to Git. The 13-episode inventory is saved on the original server at
`/tmp/uniretarget-final-recon-inventory.json` (the current two jobs use its episode-1 subset); preparation copies it into the run root.
For another server, transfer the reference bundles and update the inventory paths,
or transfer a prepared root while preserving its absolute path (its manifests and
runtime references bind that path).

```bash
python3 robotic_grounding/flash_chord/experiments/online_critic/prepare_batch.py \
  --inventory /absolute/path/to/episode-inventory.json \
  --priors-root /absolute/path/to/priors \
  --kit /absolute/path/to/v2d_submission_kit \
  --run-root "$RUN_ROOT"
bash "$RUN_ROOT/run.sh" preflight
python3 robotic_grounding/flash_chord/experiments/online_critic/submit_batch.py \
  --run-root "$RUN_ROOT" --partition YOUR_GPU_PARTITION --concurrency 3
```

CPU preflight checks the selected episodes against the public object/frame roster, asset closure,
resolved smoke/full configurations and controller settings, and runs regression tests.
It binds the passed test report to the frozen source, launcher and data manifests.
The submitted arrays use seed 43 once per episode, with at most three simultaneous
GPU jobs. All episodes first receive 1,048,576 main transitions and a real unassisted
full-horizon export. After that array finishes, separate fresh 100,663,296-transition
runs start for episodes that passed their integration gate. These are independent
policies per episode; this is not a shared multi-episode policy.

The first collection prioritizes validated coverage of the selected inputs, not converged
policies. Available `final_recon` episodes: 1, 3, 5, 6, 8, 10, 19, 32, 33, 35, 36, 37, 38. Every reference
comes from `final_recon`; older prepared inputs are excluded. Runnable reconstruction
inputs do not establish physical success. The exact selected inventory and missing
public episodes are recorded in `batch_manifest.json`.

Each exporter loads the archived best deployable checkpoint and its matching reference,
selected by measured unassisted successful-prefix length and then physical cost.
It writes actual one-world trajectories and verifies frame count, object names, finite
poses, unit quaternions and provenance. No predicted or synthetic result fills a missing
episode. The official packer produces `submissions/track3_smoke.csv` only when all 20
validate. Later, `submissions/track3_pilot.csv` uses full-run exports with valid first-pass
fallbacks for missing full results. The adjacent `.readiness.json` identifies any missing,
invalid or fallback episode. For a 13-episode batch, packaging verifies every selected output and writes
a not-ready combined-submission report listing the seven missing episodes (14, 15,
16, 20, 25, 28, 30); it does not fabricate a complete CSV. These jobs prepare files;
they do not upload a submission.

See [submission contract](submission/README.md). Job IDs are saved to
`reports/submitted_jobs.json`. Each episode's `stage.json` and Slurm logs show progress.

## Efficiency and stability review

Critic sampling now builds its eligible/train/held-out pools once per fit and transfers
loss summaries to the host once per burst. Regression tests preserve the old sampling
sequence. Checkpoint media run every 5,017,600 main transitions in full training;
validated encodes release raw intermediates, and completed overlays are reused.

Finite-response validation remains expensive: a round can add up to 4,587,520 lower
transitions (one reused control plus four candidate branches), versus 501,760 main
transitions between rounds. Count adaptation transitions and wall time separately when
reporting efficiency. A historical run spent about 2.21 adaptation transitions per main
transition with no accepted edits; additional critic fitting is not yet evidence of better
end-to-end efficiency. Per-fit held-out TD rows are correlated calibration evidence,
not an independent generalization test. Full joint training resume remains unsupported.
