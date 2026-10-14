# UniRetargeting

Refine robot reference trajectories using feedback from a physics-trained policy.
The current Track 3 experiment uses two floating Sharpa hands, FlashSAC at the lower
level, and adaptive local corrections to both wrists and finger joints at the upper
level. Five prepared episodes and their object/collision assets are included through
Git LFS, so training can run on another GPU server without the reconstruction folders.

**[Portable setup and launch instructions](robotic_grounding/uniretarget/RUNNING.txt)**

```bash
git lfs install
git lfs pull
docker build -f robotic_grounding/flash_chord/docker/Dockerfile \
  -t flash_chord:chord100m-base robotic_grounding/flash_chord
docker build -f robotic_grounding/uniretarget/Dockerfile.media \
  -t uniretarget:chord100m robotic_grounding/uniretarget
python3 robotic_grounding/uniretarget/launch.py --episode 1 --seed 43 --gpu 0
```

Configure `WANDB_MODE=online`, `WANDB_API_KEY`, `WANDB_ENTITY`, and `WANDB_PROJECT`
for online logging; the portable launcher defaults to offline logging. See the guide
for Slurm, input preparation, fresh output directories and the required GPU driver.

| Setting | Current experiment |
| --- | --- |
| Episodes | 000001, 000003, 000005, 000006, 000035 |
| Seeds | 43, 44, 45 |
| Full training | 100,663,296 environment steps; 2048 worlds |
| Upper search | Every 501,760 steps; 2048 candidates |
| Upper optimizer ablation | Independent Gaussian proposals vs learned FlashSAC proposals |
| Reference action | 56D local increments: both wrists and all finger joints |
| Window selection | Persistent object error, earlier error onset, and sampled history |
| Upper objective | Object motion and CHORD-style wrench support, plus stability/edit penalties |
| Lower response per candidate | Solved regularized policy-head surrogate; frozen critic/backbone during search |
| Validation | Fresh paired resets; every object and successful prefix protected |
| Physics | Pot 200g, lid/spoon 100g; s44 wrist controller settings |

An accepted update publishes the reference and adapted policy head together. Replay
is cleared when the reference changes. After a rejected edit, replay remains and the
adapted no-edit head is retained. VOC starts at 1 and decreases through a performance
gate. Each full launch first runs an integration smoke in the same GPU allocation.

The explicit per-stage lower problem is strongly convex and has a checked numerical
residual bound. This does not establish global convergence of neural SAC or of the
complete alternating loop. The upper evaluates a finite candidate population, proposed by independent Gaussian
sampling or a learned FlashSAC actor (`--upper gaussian` / `--upper flashsac`).
Both retain the same candidate-specific lower solve and acceptance gates.
[Equations and guarantee scope](robotic_grounding/flash_chord/docs/response_bilevel.txt).

W&B records training, accepted corrections, historical checkpoint videos, and original
hands overlaid at 45% opacity. Six Track 3 curves use reconstructed poses as proxy ground
truth under `track3_estimated/*`, evaluated with VOC=0. Assisted scores are separate.
CD-O is a labelled mesh self-comparison (zero); hidden-scan reconstruction accuracy
cannot be estimated from object poses alone. Single-object RPE is also zero by definition.

| Code | Purpose |
| --- | --- |
| [Portable workflow](robotic_grounding/uniretarget/) | Inputs, preparation, Docker and Slurm launchers |
| [Response editor](robotic_grounding/flash_chord/src/flash_chord/retargeting/response_editor.py) | Candidate-specific lower responses and paired acceptance |
| [Adaptive windows](robotic_grounding/flash_chord/src/flash_chord/retargeting/failure_windows.py) | Failure/onset detection and window selection |
| [Convex lower response](robotic_grounding/flash_chord/src/flash_chord/retargeting/response_bilevel.py) | Ridge solve and residual/error bounds |
| [Track 3 estimates](robotic_grounding/flash_chord/src/flash_chord/evaluation/track3_estimates.py) | Six-metric checkpoint reporting and proxy assumptions |
| [Historical guide](robotic_grounding/flash_chord/UNIRETARGETING.md) | Earlier Vega/tissue-box method and experiments |

The queued 100.7M batch passed 102 targeted CPU tests; full GPU integration remains gated
by each job's smoke. Current experiments have not yet established improved task success.
