# UniRetarget: Unify Kinematics & Physics Retargeting via Bi-Level Optimization

UniRetarget refines a reconstructed robot reference using feedback from a physics-trained
controller. The current implementation uses **lower FlashSAC training**, an **upper actor
with a TD wrench-cost critic**, and local corrections to both wrists and all finger joints.
The critic also learns from measured candidate/control rollouts, including rejected edits.
The current `wrench-feedback-fast` recipe bounds validation work while retaining every
selected window, rather than ranking windows and discarding proposals before testing.
Five prepared Track 3 episodes, object meshes, collision assets and robot assets are included
through Git LFS.

- **[Run on another GPU server](robotic_grounding/uniretarget/RUNNING.txt)** — setup, W&B, smoke runs, full training and Slurm.
- **[Instructions to give another agent](robotic_grounding/uniretarget/AGENT_HANDOFF.txt)** — reproduce the current three-seed smoke experiment.
- **[Method, equations and limitations](robotic_grounding/flash_chord/docs/wrench_hypergradient.txt)**.

From a clean checkout of this repository's current `main`:

```bash
git lfs install --local
git lfs pull --include='robotic_grounding/uniretarget/**,robotic_grounding/flash_chord/src/flash_chord/assets/**' --exclude=''
docker build -f robotic_grounding/flash_chord/docker/Dockerfile \
  -t flash_chord:chord100m-base robotic_grounding/flash_chord
docker build -f robotic_grounding/uniretarget/Dockerfile.media \
  -t uniretarget:chord100m robotic_grounding/uniretarget

python3 robotic_grounding/uniretarget/launch.py \
  --upper wrench_hypergradient --prepare-only --gpu 0
python3 robotic_grounding/uniretarget/launch.py \
  --upper wrench_hypergradient --episode 1 --seed 43 --phase smoke --gpu 0
```

Replace `0` with your allocated GPU index or UUID. The pinned image requires a compatible
NVIDIA driver (>=580), Docker and NVIDIA Container Toolkit. The host needs Python 3 and
Git LFS; scientific Python dependencies run inside Docker. The guide covers authentication
and a fresh output root. **Always specify `--upper wrench_hypergradient`: omitting it
selects the older Gaussian-search engine.**

For W&B, set `WANDB_MODE=online`, `WANDB_API_KEY`, `WANDB_ENTITY=DexDreamer` and
`WANDB_PROJECT=v2d-track3-uniretarget` in the launch environment. Otherwise the portable
launcher logs offline. It does not automatically mount the host's W&B login file.

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

The upper differentiates through three temporary lower actor learning updates to propose
an edit. Bounded, complete FlashSAC adaptation then checks that edit in physics against
an equally trained unchanged control. Accepted edits publish the reference and the complete
adapted lower learning state together; rejected edits retain the incoming pair and replay.
At each adaptation checkpoint, the unassisted evaluation's actual costs train the upper
critic, even when the reference is rejected. Complete rollout segments provide discounted
return targets alongside branch-matched TD targets. Branch observations stay paired with
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
Passing integration does not establish improved task performance or require accepting an edit.

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
