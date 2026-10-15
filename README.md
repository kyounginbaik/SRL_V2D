# UniRetarget: Unify Kinematics & Physics Retargeting via Bi-Level Optimization

UniRetarget refines a reconstructed robot reference using feedback from a physics-trained
controller. The current implementation uses **lower FlashSAC training**, an **upper actor
with a TD wrench-cost critic**, and local corrections to both wrists and all finger joints.
The critic also learns from measured candidate/control rollouts, including rejected edits.
The current `wrench-adam-feedback` recipe bounds validation work while retaining every
selected window, rather than ranking windows and discarding proposals before testing.
Five prepared Track 3 episodes, object meshes, collision assets and robot assets are included
through Git LFS.

- **[Run on another GPU server](robotic_grounding/uniretarget/RUNNING.txt)** — setup, W&B, smoke runs, full training and Slurm.
- **[Instructions to give another agent](robotic_grounding/uniretarget/AGENT_HANDOFF.txt)** — reproduce the current three-seed full training experiment.
- **[Method, equations and limitations](robotic_grounding/flash_chord/docs/wrench_hypergradient.txt)**.

The current experiment is **episode 1, seeds 43/44/45**, using the current `main`
implementation with Adam response updates and measured-return calibration. Each job runs a full-size GPU integration check, then automatically
starts fresh **100,663,296-step training** if validation passes. The commands below
run the same method on another server without the original `/data1` directories.

Use Linux, Docker with NVIDIA Container Toolkit, a compatible NVIDIA driver (>=580),
host Python 3 and Git LFS. The submitted jobs request **one GPU, 12 CPUs and 96 GB
host RAM per seed**; the cluster has L40S 48 GB and B200 GPUs. Scientific Python
dependencies are installed inside Docker.

1. Clone and build from the repository root:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --branch main \
  https://github.com/kyounginbaik/SRL_V2D.git SRL_V2D_uniretarget
cd SRL_V2D_uniretarget
git rev-parse HEAD
git lfs install --local
git lfs pull --include='robotic_grounding/uniretarget/**,robotic_grounding/flash_chord/src/flash_chord/assets/**' --exclude=''
docker build -f robotic_grounding/flash_chord/docker/Dockerfile \
  -t flash_chord:chord100m-base robotic_grounding/flash_chord
docker build -f robotic_grounding/uniretarget/Dockerfile.media \
  -t uniretarget:chord100m robotic_grounding/uniretarget
```

2. Choose a fresh output directory and configure logging:

```bash
export UNIRETARGET_RUN_ROOT="$PWD/.uniretarget/wrench-adam-feedback-full-run1"
export UNIRETARGET_IMAGE=uniretarget:chord100m
export WANDB_MODE=online
export WANDB_ENTITY=DexDreamer
export WANDB_PROJECT=v2d-track3-uniretarget
export WANDB_BASE_URL=https://api.wandb.ai
```

Provide `WANDB_API_KEY` through the destination server's environment or secret store
before launching. The launcher forwards that variable; it does not mount the host's
W&B login file. Use an entity/project you can write to. For local-only logging set
`WANDB_MODE=offline`. Leave `UNIRETARGET_RUN_TAG` unset to generate distinct run IDs
from the hostname and output root.

3. Prepare all bundled inputs, without allocating a GPU:

```bash
python3 robotic_grounding/uniretarget/launch.py \
  --upper wrench_hypergradient --prepare-only --gpu 0
```

4. Start the validated full run on an allocated GPU. Replace `GPU_UUID_OR_INDEX`
with its actual device UUID or index:

```bash
python3 robotic_grounding/uniretarget/launch.py \
  --upper wrench_hypergradient --episode 1 --seed 43 --phase full \
  --gpu GPU_UUID_OR_INDEX
```

Repeat with seeds **44 and 45**, either sequentially on one allocated GPU or on separate
allocated GPUs. Use a persistent terminal session for direct launches. Episodes
**3, 5, 6 and 35** are also bundled; change `--episode` to select one.

For **Slurm**, use this alternative to the direct launch. Submit from the repository
root after exporting the variables above; replace `YOUR_PARTITION` and add any required
account/QoS options. The script selects the allocated GPU automatically:

```bash
mkdir -p slurm_logs
for seed in 43 44 45; do
  sbatch --export=ALL --partition=YOUR_PARTITION --time=7-00:00:00 \
    --job-name="wff-e1-s${seed}" \
    --output="slurm_logs/%j.out" --error="slurm_logs/%j.err" \
    robotic_grounding/uniretarget/train.sbatch 1 "$seed" wrench_hypergradient full
done
```

The Slurm launcher checks GPU indices against device minor numbers. If the mapping is
ambiguous, it stops with instructions to set `UNIRETARGET_GPU_ID_KIND=minor` or `index`
according to the site's Slurm configuration. The original node01/node100/node200 cluster
uses `minor`; export that setting before `sbatch` there. An explicit
`UNIRETARGET_GPU_UUID` can disambiguate a device within the allocation.

The seven-day time limit matches the submitted full jobs; it is not a measured runtime
estimate. Choose a limit supported by the destination scheduler. Each job validates
its own seed before continuing to full training. Do not launch the same seed both
directly and through Slurm. **Always specify `--upper wrench_hypergradient` (or its
Slurm positional argument): the default selects the older Gaussian-search engine.**

To inspect the Docker command without training, add `--dry-run` to a direct launch.
All commands and phase options are described in [RUNNING.txt](robotic_grounding/uniretarget/RUNNING.txt).

| Setting | Smoke | Full training (`pilot` output phase) |
| --- | --- | --- |
| Training worlds | 2,048 | 2,048 |
| Main environment transitions | 524,288 | 100,663,296 |
| Upper update interval | 501,760; one round | 501,760; about 200 rounds |
| Upper proposals | One per selected window, up to four | Same |
| Upper action | Local 56D correction: 12 wrist + 44 finger dimensions | Same |
| Adaptation checkpoints, per branch | Three, chosen from the scoring interval; at most 448 vector steps | Same |
| Checkpoint/media interval | Final checkpoint | 1,003,520 plus final checkpoint |

Adaptation transitions are additional to the main training budget. Each comparison trains
an unchanged-reference control and candidate controllers from the same incoming learner
on fresh experience. The main run pauses during this work. The control curve is reused
across up to four candidate windows. Early windows can use 64 / 96 / 128 vector steps
(131,072 / 196,608 / 262,144 transitions per branch). Later windows receive more time to
reach the scoring interval, with a strict 448-vector-step cap (917,504 transitions).
The cap permits reaching episode 1's final frames; readiness still checks actual coverage.
Runtime also includes physics evaluations and media. The 2,048 worlds are training worlds,
not 2,048 upper candidates. This profile replaces the older full-horizon 1 / 2 / 4-pass budget.

`--phase smoke` stops after integration validation. Once its report says `passed: true`,
use `--phase pilot` with the same episode, seed and run root to start fresh full training.
`--phase full` runs smoke then fresh full training automatically, and should only be used
when that episode/seed has not already run in the chosen root. Full training does not
continue the smoke weights. There is no complete training resume; keep the checkout
unchanged while jobs run and use a fresh checkout/root for changes or retries.
The portable launcher links the training source to the checkout; pulling new code or
changing package documentation after preparation can invalidate an existing run root.

Check `$UNIRETARGET_RUN_ROOT/reports/smoke_validation_episode_000001_s43.json` for
`passed: true`. Full results are saved under:

```text
$UNIRETARGET_RUN_ROOT/episode_000001/runs/pilot/episode_000001/chunk/seed_43/
  training/upper_updates.jsonl        accepted/rejected edits, windows and timing
  training/critic_feedback.jsonl      measured feedback from every tested branch
  training/adaptation_progress.jsonl  additional candidate/control training
  training/best_demo.json             selected saved policy/reference pair
  wandb_sync/run.json                 actual W&B URL
```

Full-run W&B names begin `wrench-adam-feedback-full-d56-w2048-pilot-episode_000001-s`.
Validation has a separate `smoke` run. Queued jobs do not appear in W&B until started.
View `videos/physics_checkpoint`, `videos/reference_overlay`, `reference_change/*`
and `upper_feedback/*` to inspect motion changes and critic learning.

The upper differentiates through three temporary lower actor Adam updates to propose
an edit, carrying the incoming optimizer moments and learning-rate schedule. A tiny
variance regularizer keeps the derivative finite at zero Adam moments; the lower
critic, running statistics and sampled state distribution remain fixed in this approximation. Bounded, complete FlashSAC adaptation then checks that edit in physics against
an equally trained unchanged control. Accepted edits publish the reference and the complete
adapted lower learning state together; rejected edits retain the incoming pair and replay.
Before proposing edits, the fresh current-controller baseline calibrates the upper critic.
Proposal validation uses only this round's baseline states. At each adaptation checkpoint,
the unassisted evaluation's actual costs train the upper critic, even when the reference is
rejected. Complete rollouts provide direct measured discounted-return targets. Ordinary
replay still uses TD learning; both targets share a fixed 0.01 scale. If TD/return blending
is enabled, targets are blended before the Huber loss so opposing clipped gradients cannot
cancel. Raw and normalized diagnostics are logged separately. Branch observations stay paired with
their exact reference and lower actor; they are not relabeled as accepted-reference data
or mixed into the main actor replay. This is a finite-step bilevel approximation.
Physical adaptation is validation, outside the
gradient calculation; no full neural-RL convergence guarantee is claimed.

Both hands share a fixed demonstration start. Window selection uses object failure and
earlier error onset/history. Object targets stay fixed. Pot mass is 200 g; lid/spoon mass
is 100 g. VOC starts at 1 and decreases through performance gates. Main evaluation uses
VOC=0. W&B includes gradient/acceptance metrics, wrist and finger edit sizes, historical
physics videos, and original/current reference overlays at 45% original opacity.
Six `track3_estimated/*` metrics use reference poses as proxy ground truth; CD-O is mesh
self-comparison, not a measurement against the hidden scan.

Validate the new smoke report on your server. It now requires measured feedback from
every tested branch, actual upper-critic parameter changes, and adaptation within the
configured cap, in addition to finite learning, fixed starts and checkpoint videos.
Early training cutoffs with stale next-reference times are excluded from upper TD fitting;
complete evaluations supply the actual continuation costs. Success stability accounts for one frame of the scored interval;
object-pose and prefix guards still apply. Passing integration does not establish improved
task performance or require accepting an edit.

| Code | Purpose |
| --- | --- |
| [Portable workflow](robotic_grounding/uniretarget/) | Included inputs, Docker, W&B and Slurm launchers |
| [Upper optimizer](robotic_grounding/flash_chord/src/flash_chord/retargeting/wrench_hypergradient.py) | Upper actor, TD cost critic and finite-step response gradient |
| [Refinement editor](robotic_grounding/flash_chord/src/flash_chord/retargeting/wrench_hypergradient_editor.py) | Window proposals, physical comparison and publication |
| [Lower adaptation](robotic_grounding/flash_chord/src/flash_chord/retargeting/lower_adaptation.py) | Matched full FlashSAC training on fresh data |
| [Track 3 estimates](robotic_grounding/flash_chord/src/flash_chord/evaluation/track3_estimates.py) | Six-metric reporting and proxy assumptions |

The older `--upper gaussian` and `--upper flashsac` candidate-search experiments remain
available. Their [convex-response formulation](robotic_grounding/flash_chord/docs/response_bilevel.txt)
is a separate method; its numerical residual bounds do not apply to the new neural response.
