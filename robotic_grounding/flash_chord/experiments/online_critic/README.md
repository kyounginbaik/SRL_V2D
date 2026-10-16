# Episode-1 online-critic training

This is the 2026-10-16 pot-and-lid experiment with two floating Sharpa hands.
Preparation snapshots source, launchers, reference data and the exploration prior
into a fresh directory. Each launch checks SHA-256 manifests before starting.

## Clone and build

Use Linux, host Python 3.11+, Docker with NVIDIA Container Toolkit, and Git LFS.
The pinned image uses CUDA 13 JAX wheels and requires NVIDIA driver >=580.
The production allocation is one 48 GiB-class GPU, 12 CPUs and 96 GiB host RAM;
the container limit is 90 GiB. Allow substantial storage for checkpoints and videos.

```bash
git clone --branch uniretarget/online-critic-full-training \
  git@github.com:exaFLOPs26/SRL_V2D.git
cd SRL_V2D
git lfs install
git lfs pull
cd robotic_grounding/flash_chord
docker build -f docker/Dockerfile --target dev -t flash_chord:uniretarget-base .
docker build -f experiments/online_critic/Dockerfile \
  -t flash_chord:uniretarget-online .
cd ../..
```

The second image adds pinned OpenCV for checkpoint overlays without replacing numpy.
On the original server, set `UNIRETARGET_IMAGE=flash_chord:mixed-ep01-media` to use
the tested existing image. Runtime code comes from the prepared snapshot.

## Transfer inputs

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
`smoke` runs only integration. `pilot` starts the full budget directly and bypasses
the integration gate; use `full` on a new server. Do not restart a trained seed/phase:
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
is 56 local correction coefficients. The cost critic receives lower observations,
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
| `evaluation/metrics.json` | Final unassisted evaluation. |

The run root saves `reports/smoke_validation_episode_000001_sN.json` with actual
integration results. Logs go to the terminal or Slurm paths. Share reference snapshots
and metadata with policy checkpoints.

More critic fitting is not yet proven to improve task success. Paired diagnostics
compare predictions/returns on realized trajectories, not the fixed-state hypergradient.
Fixed-start evaluation does not establish generalization to other resets or episodes.
