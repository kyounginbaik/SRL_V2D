# UniRetargeting

UniRetargeting learns robot reference corrections using feedback from a physics-trained
motor policy. The lower level is FlashSAC. The current upper level differentiates
through a short lower-actor learning response to propose local edits, then validates
each edit with actual candidate/control training in Newton physics.

**[Run the current experiment on another server](robotic_grounding/flash_chord/experiments/online_critic/README.md)**
provides container builds, input transfer, verification, direct GPU launch, Slurm
submission, and output inspection.

The current recipe uses episode 1's pot-and-lid motion, two floating Sharpa hands,
2,048 training worlds, and seeds 43–45. Each full job runs a 1,048,576-transition
integration check, verifies online critic fitting between two upper rounds, then
starts fresh training for **100,663,296 main transitions**. Candidate/control adaptation
and evaluation add simulation work beyond that budget. Full bilevel resume is unsupported.

## Current algorithm

| Component | Current recipe |
| --- | --- |
| Upper actor input | Reference statistics, current correction summaries, temporal signed hand-tracking errors, execution quality, selected-window and failure/onset timing, and local per-object pose errors. |
| Upper actor output | 56 bounded coefficients: wrist translation/rotation and 22 finger joints per hand, applied through a smooth local time window. |
| Upper critic input | Reference-conditioned lower observations, critic context, and lower robot actions. It estimates discounted wrench-support cost, not a Q-function over upper-edit coefficients. |
| Proposal gradient | Three differentiable lower-actor Adam steps with incoming moments; lower critics and statistics stay fixed. Four upper optimizer steps produce one proposal per selected window. |
| Critic fitting | Outer-round fitting and measured unassisted-return fitting, plus four TD updates every 16 main vector steps after the first upper round, using valid current-reference/current-generation training rows. |
| Physical validation | Matched candidate/control FlashSAC adaptation capped at 448 vector steps per branch, with object, prefix and improvement checks followed by frozen confirmation. |
| Upper schedule | Every 501,760 main transitions; up to four selected windows. |

The additional critic updates are an experiment; improved retargeting performance
remains unproven. Differentiation covers a short actor response and learned surrogate,
not the simulator or converged RL training.

## Run it

Use the [complete cross-server guide](robotic_grounding/flash_chord/experiments/online_critic/README.md).
You need Linux, Docker with NVIDIA GPU support, materialized Git LFS robot assets,
and the separate checksummed episode input bundle. The launch requests one GPU,
12 CPU cores and 96 GiB host memory. Budget a 48 GiB-class GPU. The image uses CUDA
13 JAX wheels; its Dockerfile requires an NVIDIA driver at least 580.

After building the image and obtaining the inputs:

```bash
python3 robotic_grounding/flash_chord/experiments/online_critic/prepare.py \
  --data-root /absolute/path/to/episode-inputs \
  --run-root /absolute/path/to/new-run
bash /absolute/path/to/new-run/run.sh preflight
GPU_DEVICE=0 bash /absolute/path/to/new-run/run.sh full 43
```

W&B is optional and disabled by default. The launcher accepts arbitrary absolute run
paths, without requiring the original server's directories, username or hostname.
The same prepared run supports seeds 43, 44 and 45 on separate GPUs.

## Code and evidence

| File | Purpose |
| --- | --- |
| [Training entry](robotic_grounding/flash_chord/scripts/train_flash_sac.py) | Scene, lower learner, upper editor, and curriculum. |
| [Upper editor](robotic_grounding/flash_chord/src/flash_chord/retargeting/wrench_hypergradient_editor.py) | Online critic schedule, physical trials, diagnostics, and publication. |
| [Upper learner](robotic_grounding/flash_chord/src/flash_chord/retargeting/wrench_hypergradient.py) | Cost critics, upper actor, and differentiable response objective. |
| [Lower adaptation](robotic_grounding/flash_chord/src/flash_chord/retargeting/lower_adaptation.py) | Isolated, matched full FlashSAC candidate/control training. |
| [Method details](robotic_grounding/flash_chord/docs/wrench_hypergradient.txt) | Objective, adaptation budgets, acceptance and limitations. |
| [Critic diagnostics](robotic_grounding/flash_chord/docs/critic_diagnostics_20261016.txt) | Historical audit and interpretation of the online-critic experiment. |

Runs save lower policies, reference versions, upper decisions, online critic updates,
complete trial trajectories, critic snapshots and checkpoint videos. Copy matching
references and metadata with any shared policy checkpoint.

The [older tissue-box guide](robotic_grounding/flash_chord/UNIRETARGETING.md) describes
the earlier TTSA/FlashSAC upper editor. Its 72-dimensional action, Vega-arm embodiment
and schedule do not describe this recipe.
